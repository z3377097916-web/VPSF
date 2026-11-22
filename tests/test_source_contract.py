"""Synthetic contract tests; no generated model evidence is fabricated for training."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from test_fix2 import raw_record
from preference.protocol import POLICY,PROMPT,FEEDBACK_BOUNDARY
from preference.source_contract import raw_source,fingerprint
from preference.purity import RULE,evaluate
from scripts import extract_features,prepare_pairs,run_training
from rlsfv.scoring import score


class SourceContractTests(unittest.TestCase):
    def export(self,root,record,diagnostic=False):
        raw=root/'raw.jsonl';raw.write_text(json.dumps(record)+'\n')
        out=root/'features';args=['features','--input',str(raw),'--output',str(out)]
        if diagnostic:args.append('--diagnostic')
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):extract_features.main()
        return out

    def prepare(self,root,features,diagnostic=False):
        out=root/'pairs';args=['prepare','--data',str(features),'--models',str(ROOT/'models'),
            '--output',str(out),'--expected-images','1']
        if diagnostic:args.append('--diagnostic')
        with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):prepare_pairs.main()
        return out

    def test_source_export_prepare_and_training_validation(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=raw_record()
            d['facts']['dog']['checks']=[{'status':'supported'},{'status':'supported'}]
            d['ranking']=score(d['candidates'],d['facts'],json.loads((ROOT/'configs/generation.json').read_text())['scoring'],42)
            pairs=self.prepare(p,self.export(p,d))
            prov=json.loads((pairs/'provenance.json').read_text())
            run_training.validate_pair_source(dict(data_policy=POLICY,purity_rule=RULE),prov)
            rows=json.loads((pairs/'E3_combined.json').read_text());self.assertTrue(rows)
            run_training.validate_rows(rows,prov)
            self.assertEqual(prov['source_contract']['prompt'],PROMPT)
            self.assertEqual(prov['source_contract']['model']['variant'],'base')
            self.assertIn('feature_contract.json',{Path(k).name for k in prov['inputs']})
            cfg=json.loads((ROOT/'configs/run.json').read_text())
            cfg.update(model_path='/server/models/base',vision_path='/server/models/vision',
                output_root='/server/e3_final/models',evaluation_manifest='/server/amber_images.json',expected_source_images=1)
            config=p/'run.json';config.write_text(json.dumps(cfg));printed=io.StringIO()
            with patch.object(sys,'argv',['train','--config',str(config),'--pairs',str(pairs)]),contextlib.redirect_stdout(printed):
                run_training.main()
            tokens=printed.getvalue().split()
            self.assertIn('--nproc_per_node',tokens)
            self.assertEqual(int(tokens[tokens.index('--nproc_per_node')+1]),4)
            self.assertIn('--num_train_epochs 2',printed.getvalue())

    def test_long_actual_prompt_rejected_before_feature_publication(self):
        with tempfile.TemporaryDirectory() as td:
            d=raw_record();d['candidates'][0]['prompt']='Describe this image in one complete paragraph.'
            with self.assertRaisesRegex(ValueError,'Actual fresh'):self.export(Path(td),d)
            self.assertFalse((Path(td)/'features').exists())

    def test_later_fresh_and_common_scoring_prompts_checked(self):
        for which in ('later_fresh','fuzzy'):
            d=raw_record()
            if which=='later_fresh':d['candidates'][-1]['prompt']='Different prompt'
            else:d['candidates'][5]['fuzzy']['common_prompt']='Different prompt'
            with self.assertRaises(ValueError):raw_source(d)

    def test_missing_identity_rejected(self):
        d=raw_record();del d['generation_identity']
        with self.assertRaisesRegex(ValueError,'identity'):raw_source(d)

    def test_complete_generation_directory_exports_formal_contract(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=raw_record();run=p/'candidates';folder=run/'images/sample';folder.mkdir(parents=True)
            (folder/'result.json').write_text(json.dumps(d))
            (run/'run.json').write_text(json.dumps(dict(d['generation_identity'],records=[d['record']])))
            (run/'status.json').write_text(json.dumps(dict(complete=True)))
            out=p/'features';args=['features','--input',str(run),'--output',str(out)]
            with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):extract_features.main()
            contract=json.loads((out/'feature_contract.json').read_text())
            self.assertEqual(contract['usage'],'training');self.assertEqual(contract['images'],1)
            self.assertEqual(contract['source']['prompt'],PROMPT)

    def test_different_backend_adapter_or_checkpoint_rejected(self):
        for case in ('backend','adapter','weight','revision','missing_weight'):
            d=raw_record();identity=d['generation_identity'];meta=identity['model']
            if case=='backend':identity['config']['backend']='qwen3'
            if case=='adapter':meta['variant']='trained';meta['adapter_sha256']={'adapter':'0'*64}
            if case=='weight':meta['model_inventory'][0]['sha256']='0'*64
            if case=='revision':meta['model_inventory'][0]['revision']='another-revision'
            if case=='missing_weight':meta['model_inventory'].pop()
            with self.subTest(case=case),self.assertRaises(ValueError):raw_source(d)

    def test_mixed_run_identity_rejected(self):
        d=raw_record();other=copy.deepcopy(d['generation_identity']);other['config']['seed']=1
        with self.assertRaisesRegex(ValueError,'run.json'):raw_source(d,other)

    def test_diagnostic_export_cannot_become_training_pairs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);d=raw_record();del d['generation_identity']
            features=self.export(p,d,diagnostic=True)
            with self.assertRaisesRegex(ValueError,'Diagnostic'):self.prepare(p,features)
            pairs=self.prepare(p,features,diagnostic=True)
            prov=json.loads((pairs/'provenance.json').read_text())
            with self.assertRaisesRegex(ValueError,'Diagnostic'):
                run_training.validate_pair_source(dict(data_policy=POLICY,purity_rule=RULE),prov)

    def test_feature_edits_after_export_are_detected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);features=self.export(p,raw_record())
            f=features/'candidate_features.json';data=json.loads(f.read_text());data[0]['candidates'][0]['text']='changed'
            f.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'changed after export'):self.prepare(p,features)

    def test_per_pair_prompt_policy_schema_and_source_are_checked(self):
        source,_=raw_source(raw_record());prov=dict(source_contract_hash=fingerprint(source))
        row=dict(prompt=PROMPT,data_policy=POLICY,feature_schema='legacy_coco_v1',
                 source_contract_hash=fingerprint(source),purity_rule=RULE,
                 purity_checks=[evaluate({'p':.8},{'p':.6})])
        run_training.validate_rows([row],prov)
        for key in ('prompt','data_policy','feature_schema','source_contract_hash','purity_rule'):
            with self.subTest(field=key),self.assertRaises(ValueError):
                run_training.validate_rows([dict(row,**{key:'incorrect'})],prov)

    def test_purity_boolean_cannot_hide_wrong_p_direction(self):
        source,_=raw_source(raw_record());h=fingerprint(source)
        row=dict(prompt=PROMPT,data_policy=POLICY,feature_schema='legacy_coco_v1',source_contract_hash=h,purity_rule=RULE)
        for pc,pr in ((.6,.8),(.8,.8),(None,.6)):
            row['purity_checks']=[dict(passed=True,chosen_p=pc,rejected_p=pr)]
            with self.assertRaisesRegex(ValueError,'strict p'):run_training.validate_rows([row],dict(source_contract_hash=h))

if __name__=='__main__':unittest.main()
