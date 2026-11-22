import copy,contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from preference.inputs import read_bundle
from preference.purity import evaluate
from preference.selectors import choose,fallback_counts
from preference.protocol import POLICY,PROMPT,FEEDBACK_BOUNDARY
from scripts import prepare_pairs
from scripts.pipeline import feature_source
from scripts.run_training import validate_pair_source,validate_approved_scheme
from scripts.make_training_manifest import convert
from preference.source_contract import VERSION,expected_model,fingerprint

class E3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ims,cls.af,cls.cs,cls.cs2,_=read_bundle(ROOT/'tests/fixtures')
        cls.risk=json.loads((ROOT/'models/results_v1_2.json').read_text())['risk_model']
        cls.reject=json.loads((ROOT/'models/hc_reject_models.json').read_text())
    def test_purity_direction_and_missing_values(self):
        c=lambda p,m:dict(p=p,coverage=dict(f=m,s=m))
        self.assertTrue(evaluate(c(.8,.5),c(.6,.5))['passed'])
        self.assertFalse(evaluate(c(.8,.7),c(.8,.5))['passed'])
        self.assertTrue(evaluate(c(.8,.4),c(.6,.5))['passed'])
        self.assertFalse(evaluate(c(.5,.7),c(.8,.5))['passed'])
        self.assertFalse(evaluate(c(.8,.5),c(.8,.5))['passed'])
        for x in (None,float('nan'),-1,2):self.assertFalse(evaluate(c(x,.5),c(.6,.5))['passed'])
    def test_coverage_values_do_not_control_purity(self):
        for fc in (0.,.5,1.):
            for fr in (0.,.5,1.):
                c=dict(p=.8,coverage=dict(f=fc,s=fc));r=dict(p=.6,coverage=dict(f=fr,s=fr))
                self.assertTrue(evaluate(c,r)['passed'])
                r['p']=.8;self.assertFalse(evaluate(c,r)['passed'])
        self.assertTrue(evaluate({'p':.8},{'p':.6})['passed'])
    def pairing_case(self,ca,cb,ra,rb):
        im=copy.deepcopy(self.ims[0]);cs=im['candidates']
        for idx,c in enumerate(cs):
            c['text']=f'Caption number {idx}';c['p']=.1;c['coverage'].update(f=.9,s=.9)
        for idx in (ca,cb):cs[idx]['p']=.9;cs[idx]['coverage'].update(f=.1,s=.1)
        with patch.object(prepare_pairs,'choose',return_value={'v1.2':cs[ca],'product':cs[cb],'fallback':False}),patch.object(prepare_pairs,'rejected',side_effect=[cs[ra],cs[rb]]):
            return prepare_pairs.build_e3([im],self.af,self.cs,self.cs2,self.risk,self.reject)
    def test_same_chosen_two_rejects_both_retained(self):
        passed,unique,_,_,_=self.pairing_case(0,0,2,3)
        self.assertEqual(len(passed),2);self.assertEqual(len({p['chosen'] for p in passed}),1)
        self.assertEqual(len({p['rejected'] for p in passed}),2)
    def test_two_chosen_same_reject_both_retained(self):
        passed,unique,_,_,_=self.pairing_case(0,1,2,2)
        self.assertEqual(len(passed),2);self.assertEqual(len({p['chosen'] for p in passed}),2)
        self.assertEqual(len({p['rejected'] for p in passed}),1)
    def test_identical_complete_pair_only_once(self):
        passed,unique,_,_,_=self.pairing_case(0,0,2,2)
        self.assertEqual(len(unique),1);self.assertEqual(len(passed),1)
        self.assertEqual(set(passed[0]['roles']),{'v1.2','product'})
    def test_two_distinct_pairs_retained(self):
        passed,_,_,_,_=self.pairing_case(0,1,2,3)
        self.assertEqual(len(passed),2)
    def test_fallback_reaches_pair_purity(self):
        im=next(i for i in self.ims if all(c['errors'] for c in i['candidates']))
        selected=choose(im,self.risk,self.cs,self.cs2)
        self.assertTrue(selected['fallback']);self.assertIsNotNone(selected['v1.2'])
        key=fallback_counts
        self.assertEqual(key(selected['v1.2']),min(map(key,im['candidates'])))
        passed,unique,audit,pa,stats=prepare_pairs.build_e3([im],self.af,self.cs,self.cs2,self.risk,self.reject)
        self.assertTrue(unique);self.assertEqual(len(pa),len(unique));self.assertTrue(all(x['g_fallback'] for x in pa))
    def test_only_e3_and_annotations_never_opened(self):
        with tempfile.TemporaryDirectory() as td:
            outputs=[]
            for idx in range(2):
                d=Path(td)/f'data{idx}';d.mkdir()
                for p in (ROOT/'tests/fixtures').glob('*.json'):(d/p.name).write_bytes(p.read_bytes())
                if idx:(d/'coco_metrics_DIAGNOSTIC_ONLY.json').write_text('broken forbidden annotation')
                out=Path(td)/f'out{idx}'
                args=['prepare','--data',str(d),'--models',str(ROOT/'models'),'--output',str(out),'--expected-images',str(len(self.ims)),'--diagnostic']
                with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):prepare_pairs.main()
                self.assertEqual([p.name for p in out.glob('E*.json')],['E3_combined.json'])
                outputs.append((out/'E3_combined.json').read_bytes())
            self.assertEqual(*outputs)
    def test_deduplicate_then_purity_reweight(self):
        passed,unique,_,pa,stats=prepare_pairs.build_e3(self.ims,self.af,self.cs,self.cs2,self.risk,self.reject)
        self.assertEqual(len(pa),len(unique));self.assertEqual(len(passed),sum(p['passed'] for p in pa))
        keys=[(p['image_id'],p['prompt'],p['chosen'].strip(),p['rejected'].strip()) for p in unique]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertTrue(passed)
        sums={}
        for p in passed:sums[p['image_id']]=sums.get(p['image_id'],0)+p['pair_weight']
        self.assertLess(max(sums.values())-min(sums.values()),1e-10)
    def test_no_diagnostic_training_default(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td)
            with self.assertRaises(ValueError):feature_source({},w)
            (w/'features').mkdir();self.assertEqual(feature_source({},w),w/'features')
    def test_training_rejects_old_purity_protocol(self):
        cfg=dict(data_policy=POLICY,purity_rule='strict_p_coverage_diagnostic_v2')
        with self.assertRaises(ValueError):validate_pair_source(cfg,dict(policy=POLICY,feedback_boundary=FEEDBACK_BOUNDARY))
        source=dict(version=VERSION,prompt=PROMPT,model=expected_model(),feedback_backend='same_original_model')
        validate_pair_source(cfg,dict(policy=POLICY,purity_rule='strict_p_coverage_diagnostic_v2',feedback_boundary=FEEDBACK_BOUNDARY,
            usage='training',feature_schema='legacy_coco_v1',source_contract=source,source_contract_hash=fingerprint(source)))
    def test_10000_source_images_without_original_questions(self):
        records=convert(ROOT/'manifests/rlaif10k_source.jsonl',Path('/unlabeled/rlaif'))
        self.assertEqual(len(records),10000);self.assertEqual(len({r['image'] for r in records}),10000)
        self.assertTrue(all(set(r)=={'id','image'} for r in records))
        self.assertEqual(records[0]['image'],'/unlabeled/rlaif/images/1.jpg')
    def test_generation_accepts_converter_id_image_manifest(self):
        from rlsfv.data import manifest as generation_manifest
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);image=p/'1.jpg';image.write_bytes(b'not-decoded-here')
            manifest=p/'train.json';manifest.write_text(json.dumps([{'id':'rlaif-1','image':str(image)}]))
            rows=generation_manifest(manifest)
            self.assertEqual(rows[0]['split'],'train');self.assertEqual(rows[0]['id'],'rlaif-1')
    def test_pair_prompt_matches_generation_contract(self):
        from rlsfv.config import X0
        self.assertEqual(X0,PROMPT)
        passed,_,_,_,_=self.pairing_case(0,1,2,3)
        self.assertTrue(passed);self.assertTrue(all(p['prompt']==X0 for p in passed))
    def test_approved_training_scheme_is_enforced(self):
        cfg=json.loads((ROOT/'configs/run.json').read_text());validate_approved_scheme(cfg)
        for key,value in (('beta',.2),('learning_rate',1e-6),('lora_rank',128),('max_length',1024)):
            with self.assertRaises(ValueError):validate_approved_scheme(dict(cfg,**{key:value}))

if __name__=='__main__':unittest.main()
