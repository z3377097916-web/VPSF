"""Deterministic simulated backend. NEVER scientific model results."""
import json
from pathlib import Path
from .io import atomic_json, read_json, fingerprint
from .pipeline import Pipeline
from .data import export
from .diagnostics import retention

class FixtureBackend:
    def __init__(self):
        self.meta={"variant":"SIMULATED_TEST_ONLY","weights":"none"}
        self.calls=[]
        self.index=0

    def generate(self,image,prompt,settings,seed):
        self.calls.append({"image":image,"prompt":prompt,"seed":seed})
        if image is None:
            caption=json.loads(prompt.split("Caption (data, not instructions):\n",1)[1])
            facts=[]
            def fact(cat,subject,predicate,obj,entities):
                facts.append({"category":cat,"subject":subject,"predicate":predicate,"object":obj,"scope":"image",
                    "polarity":"positive","modality":"definite","subtype":"","quote":caption,
                    "entities":[{"id":fingerprint(e)[:12],"anchor":e,"qualifiers":[]} for i,e in enumerate(entities)],"ambiguous":False})
            fact("object","person","exists","",["person"])
            if "bicycle" in caption:
                fact("object","bicycle","exists","",["bicycle"])
            if "beside" in caption:
                fact("spatial","person","beside","bicycle",["person","bicycle"])
            if "pushes" in caption:
                fact("relation","person","pushes","bicycle",["person","bicycle"])
            if "tree" in caption:
                fact("object","tree","exists","",["tree"])
            if "dog" in caption:
                fact("object","dog","exists","",["dog"])
            if "park" in caption:
                fact("scene","scene","is","park",[])
            text=json.dumps({"facts":facts})
        elif "Assertion: " in prompt:
            prop=json.loads(prompt.split("Assertion: ",1)[1])
            verdict="refuted" if prop["subject"]=="dog" else "unclear" if prop["subject"]=="tree" and prompt.startswith("Inspect the relevant") else "supported"
            v={"status":verdict,"evidence":"Simulated fixture only"}
            if prompt.startswith("Inspect the relevant"): v["observation"]="Simulated visual observation"
            text=json.dumps(v)
        else:
            i=self.index%22
            captions=["A person stands beside a bicycle.","A person pushes a bicycle near a tree.",
                      "A person is in a park.","A person and a dog stand beside a bicycle."]
            text=captions[i%4]; self.index+=1
        return {"text":text,"token_ids":[],"tokens":len(text.split()),"hit_limit":False,
                "history_messages":0,"context_messages":1,"seconds":0}

    def score(self,image,prompt,text,parser,config,sampling):
        return {"u":int(fingerprint(text)[:6],16)/0xffffff,"history_messages":0,
                "score_implementation":"SIMULATED_NOT_REAL_LOGITS","tokens":len(text.split())}

def run(output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose fresh smoke output")
    cfg=read_json(Path(__file__).parents[1]/"configs/server.example.json")
    cfg["models"]="SIMULATED"; cfg["vendor"]="SIMULATED"
    output.mkdir(parents=True)

    image=output/"simulated-image.bin"; image.write_bytes(b"SIMULATED TEST IMAGE, NOT A MODEL INPUT")
    rows=[{"id":"fixture","image":str(image),"split":"train"}]
    backend=FixtureBackend()
    p=Pipeline(cfg,backend,None,output/"run")
    status=p.run(rows)
    assert status[0]["status"]=="complete",status
    original=len(backend.calls)
    assert p.run(rows)[0]["status"]=="complete"
    assert len(backend.calls)==original
    summary=export(output/"run",output/"export")
    retention(output/"run",output/"retention.json")
    result={"status":"PASS","type":"CPU_SIMULATED_PIPELINE_ONLY","captions":22,"methods":len(summary),
        "model_calls":original,"resume_added_calls":0,"gpu_inference_tested":False,"dpo_training_tested":False}
    atomic_json(output/"smoke-report.json",result)
    return result
