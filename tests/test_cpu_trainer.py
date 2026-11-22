"""Tiny real LLaVA/PEFT/Trainer path on CPU; not a 7B/CUDA/ZeRO qualification."""
import copy,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'trainer'))
import torch
from torch import nn
from peft import LoraConfig,get_peft_model
from trl import DPOConfig
from transformers import set_seed
from llava.model.language_model.llava_llama import LlavaConfig,LlavaLlamaForCausalLM
from llava_trainer import LlavaDPOTrainer
from data import DataCollatorForSupervisedDataset
from types import SimpleNamespace

class Tokenizer:
    pad_token_id=0
    def save_pretrained(self,path):pass

class Vision(nn.Module):
    num_patches=4
    def forward(self,images):return images.mean((1,2,3))[:,None,None].expand(-1,4,8)

def model():
    set_seed(7)
    cfg=LlavaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=4,max_position_embeddings=128)
    cfg.use_cache=False;cfg.mm_patch_merge_type='flat';cfg.tokenizer_padding_side='right';cfg.tokenizer_model_max_length=128
    m=LlavaLlamaForCausalLM(cfg);m.model.vision_tower=Vision();m.model.mm_projector=nn.Linear(8,32,bias=False)
    return get_peft_model(m,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],lora_dropout=0,bias='none',task_type='CAUSAL_LM'))

class Samples(torch.utils.data.Dataset):
    def __len__(self):return 4
    def __getitem__(self,i):
        return dict(chosen_input_ids=torch.tensor([1,-200,3,4,6,2]),chosen_labels=torch.tensor([-100,-100,-100,-100,6,2]),
            reject_input_ids=torch.tensor([1,-200,3,4,8,9,2]),reject_labels=torch.tensor([-100,-100,-100,-100,8,9,2]),images=torch.ones(3,2,2),pair_weight=[.5,.5,1.5,1.5][i])

class TrainerSmoke(unittest.TestCase):
    def test_real_multimodal_dpo_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m=model();before={n:p.detach().clone() for n,p in m.named_parameters()}
            args=DPOConfig(output_dir=d,use_cpu=True,report_to='none',per_device_train_batch_size=2,num_train_epochs=2,
                learning_rate=.01,save_strategy='epoch',logging_steps=1,remove_unused_columns=False,disable_tqdm=True,seed=7)
            t=LlavaDPOTrainer(model=m,args=args,dpo_beta=.1,train_dataset=Samples(),data_collator=DataCollatorForSupervisedDataset(Tokenizer()))
            t.train();self.assertEqual(t.state.global_step,4)
            changed=[n for n,p in m.named_parameters() if not torch.equal(before[n],p)]
            self.assertTrue(changed);self.assertTrue(all('lora_' in n for n in changed))
            self.assertTrue((Path(d)/'checkpoint-2/optimizer.pt').is_file());self.assertTrue((Path(d)/'checkpoint-2/adapter_model.safetensors').is_file())
            resumed=model();t2=LlavaDPOTrainer(model=resumed,args=args,dpo_beta=.1,train_dataset=Samples(),data_collator=DataCollatorForSupervisedDataset(Tokenizer()))
            t2.train(resume_from_checkpoint=str(Path(d)/'checkpoint-2'))
            self.assertEqual(t2.state.global_step,4)
            for n,p in m.named_parameters():torch.testing.assert_close(p,dict(resumed.named_parameters())[n],atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
