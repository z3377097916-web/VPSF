import copy,json,os,math
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX,IGNORE_INDEX

class LazySupervisedDataset(Dataset):
    def __init__(self,train_data_path,tokenizer,data_args,seed=42):
        self.tokenizer=tokenizer;self.data_args=data_args
        self.data=json.load(open(train_data_path,encoding="utf8"))
        if not isinstance(self.data,list) or not self.data:raise ValueError("Empty/non-list training data")
        counts=Counter(x["image_id"] for x in self.data);M=len(self.data);N=len(counts)
        seen=set()
        for x in self.data:
            if any(not isinstance(x.get(k),str) or not x[k].strip() for k in ("image_id","img_path","prompt","chosen","rejected")):raise ValueError("Malformed training pair")
            key=(x["image_id"],x["prompt"],x["chosen"],x["rejected"])
            if key in seen or x["chosen"].strip()==x["rejected"].strip():raise ValueError("Duplicate/identical pair")
            seen.add(key)
            expected=M/(N*counts[x["image_id"]])
            if abs(x["pair_weight"]-expected)>1e-10:raise ValueError("Image weight is not globally normalized")
            if not os.path.isfile(x["img_path"]):raise FileNotFoundError(x["img_path"])
            x["_chosen"]=self.encode(x["prompt"],x["chosen"])
            x["_rejected"]=self.encode(x["prompt"],x["rejected"])
    def __len__(self):return len(self.data)
    def encode(self,prompt,answer):
        if "<image>" in prompt or "<image>" in answer:raise ValueError("Reserved image marker in text")
        conv=conversation_lib.default_conversation.copy();conv.append_message(conv.roles[0],"<image>\n"+prompt);conv.append_message(conv.roles[1],None)
        prefix=tokenizer_image_token(conv.get_prompt(),self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt")
        conv.messages[-1][1]=answer
        full=tokenizer_image_token(conv.get_prompt(),self.tokenizer,IMAGE_TOKEN_INDEX,return_tensors="pt")
        if not torch.equal(prefix,full[:len(prefix)]):raise ValueError("Prompt/answer token boundary differs; no guessed mask")
        if int((full==IMAGE_TOKEN_INDEX).sum())!=1:raise ValueError("Expected exactly one image")
        patches=int(getattr(self.data_args,"image_patch_tokens",576))
        if len(full)-1+patches>self.tokenizer.model_max_length:raise ValueError("Expanded image+answer exceeds max length; no truncation")
        labels=full.clone();labels[:len(prefix)]=IGNORE_INDEX
        if not bool(labels.ne(IGNORE_INDEX).any()):raise ValueError("No answer tokens")
        return {"input_ids":full,"labels":labels}
    def __getitem__(self,i):
        x=self.data[i];processor=self.data_args.image_processor
        with Image.open(x["img_path"]) as im:image=im.convert("RGB")
        if self.data_args.image_aspect_ratio=="pad":
            w,h=image.size;canvas=Image.new("RGB",(max(w,h),max(w,h)),tuple(int(v*255) for v in processor.image_mean));canvas.paste(image,((max(w,h)-w)//2,(max(w,h)-h)//2));image=canvas
        image=processor.preprocess(image,return_tensors="pt")["pixel_values"][0]
        a,b=x["_chosen"],x["_rejected"]
        return dict(chosen_input_ids=a["input_ids"],chosen_labels=a["labels"],reject_input_ids=b["input_ids"],reject_labels=b["labels"],images=image,pair_weight=x["pair_weight"])

class DataCollatorForSupervisedDataset:
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __call__(self,rows):
        out={"images":torch.stack([r["images"] for r in rows]),"pair_weight":torch.tensor([r["pair_weight"] for r in rows],dtype=torch.float32)}
        for side in ("chosen","reject"):
            ids=[r[side+"_input_ids"] for r in rows];labs=[r[side+"_labels"] for r in rows]
            out[side+"_input_ids"]=torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=self.tokenizer.pad_token_id)
            out[side+"_labels"]=torch.nn.utils.rnn.pad_sequence(labs,batch_first=True,padding_value=IGNORE_INDEX)
            width=out[side+"_input_ids"].shape[1]
            out[side+"_attention_mask"]=torch.arange(width)[None,:]<torch.tensor([len(x) for x in ids])[:,None]
        return out
