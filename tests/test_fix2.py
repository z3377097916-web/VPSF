"""Regression tests using synthetic raw checks, not image truth or model mocks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from rlsfv.scoring import score
from preference.selectors import choose,anchor_score,fallback_counts,reject_features
from preference.inputs import read_bundle
from scripts import extract_features,prepare_pairs
from preference.protocol import PROMPT
from preference.source_contract import expected_model


def raw_record(iid=7):
    def fact(name,low_mass=False):
        return dict(category='object',subject=name,predicate='exists',object='',scope='',
            polarity='positive',modality='definite',ambiguous=False,subtype='',quote='A dog.',
            entities=[dict(id=name,key=name,head=name,anchor=name,display=name,qualifiers=[])],
            align=dict(subject=name,predicate='exists',object=''),
            counts=[10,7,5],fresh_counts=[10,1,1],fresh_totals=[10,1,1],
            checks=[dict(status='refuted',low_mass=low_mass),dict(status='refuted')])
    cs=[dict(id=f's{t}:{i}',stage=t,branch='fresh' if t==0 or i==n-1 else 'guided',
             claim_ids=['cat'],text=f'Caption stage {t} index {i}.',fuzzy=dict(u=(t*10+i+1)/40))
        for t,n in enumerate((10,7,5)) for i in range(n)]
    cs[0]['claim_ids']=['dog'];cs[0]['fuzzy']['u']=.99
    facts={'dog':fact('dog',True),'cat':fact('cat')}
    for k,f in facts.items():
        f['counts']=[sum(k in c['claim_ids'] and c['stage']==t for c in cs) for t in range(3)]
        f['fresh_counts']=[sum(k in c['claim_ids'] and c['stage']==t and c['branch']=='fresh' for c in cs) for t in range(3)]
    cfg=json.loads((ROOT/'configs/generation.json').read_text())['scoring']
    rank=score(cs,facts,cfg,42)
    for c in cs:
        c['prompt']=PROMPT if c['branch']=='fresh' else 'Check the image, then describe it.'
        c['fuzzy']['common_prompt']=PROMPT
    identity=dict(config=dict(backend='llava',extraction=dict(mode='hybrid')),
                  model=dict(variant='base',adapter_sha256={},model_inventory=expected_model()['files']),
                  test_fixture='synthetic checks; no image/model inference')
    return dict(status='complete',record=dict(id=iid,image='/unused/7.jpg',split='train'),
                candidates=cs,facts=facts,ranking=rank,generation_identity=identity)


class Fix2Regression(unittest.TestCase):
    def test_raw_formula_feature_and_effective_g_count_are_distinct(self):
        im=extract_features.compact_record(raw_record());c=im['candidates'][0]
        self.assertEqual(c['errors'],['dog'])
        self.assertEqual(c['double_errors'],['dog'])  # Frozen R_A/R_B ABI retained.
        self.assertEqual(c['g_double_errors'],[])
        self.assertEqual(fallback_counts(c),(0,1))

    def test_low_mass_does_not_penalize_fallback_as_double_refutation(self):
        im=extract_features.compact_record(raw_record())
        sel=choose(im,None,{},{});self.assertTrue(sel['fallback'])
        self.assertEqual(sel['fallback_risk_counts'],[0,1])
        for name in ('v1.2','product'):self.assertEqual(sel[name]['id'],'s0:0')
        # The old counts tie all rows and incorrectly defer to their score.
        old=lambda c:(len(set(c['errors'])&set(c['double_errors'])),len(c['errors']))
        self.assertEqual(len({old(c) for c in im['candidates']}),1)
        self.assertNotEqual(max(im['candidates'],key=anchor_score)['id'],'s0:0')

    def test_invalid_soft_read_is_also_not_a_valid_double_refutation(self):
        d=raw_record();d['facts']['dog']['checks'][0]['low_mass']=False
        d['facts']['dog']['soft_support']={'valid':False}
        # Both cases have the same effective statuses and unchanged saved G.
        c=extract_features.compact_record(d)['candidates'][0]
        self.assertEqual(c['double_errors'],['dog']);self.assertEqual(c['g_double_errors'],[])

    def test_legacy_fallback_features_require_explicit_reexport(self):
        im=extract_features.compact_record(raw_record())
        for c in im['candidates']:del c['g_double_errors']
        with self.assertRaisesRegex(ValueError,'re-export'):choose(im,None,{}, {})

    def test_integer_image_id_is_consistent_in_all_feature_files(self):
        d=raw_record(0);im=extract_features.compact_record(d);af,c,c2=extract_features.extra(d)
        self.assertEqual(im['image_id'],'0');self.assertEqual(im['source_record']['id'],'0')
        self.assertEqual(d['record']['id'],0)  # Never mutate the source used for hashing.
        self.assertTrue(all(k.startswith('0|') for k in af))
        # This all-G-fail example has no consensus rows by design.
        self.assertEqual((c,c2),([],[]))

    def run_export(self,source,destination):
        with patch.object(sys,'argv',['extract','--input',str(source),'--output',str(destination)]),contextlib.redirect_stdout(io.StringIO()):
            extract_features.main()

    def test_failed_parse_can_be_corrected_and_retried_same_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'raw.jsonl';out=p/'features'
            src.write_text('{broken JSON\n')
            with self.assertRaises(json.JSONDecodeError):self.run_export(src,out)
            self.assertFalse(out.exists())
            src.write_text(json.dumps(raw_record())+'\n');self.run_export(src,out)
            ims,*_=read_bundle(out);self.assertEqual(ims[0]['image_id'],'7')
            self.assertEqual(len(list(out.glob('*.json'))),5)

    def test_normalized_duplicate_ids_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'raw.jsonl';out=p/'features'
            src.write_text('\n'.join(json.dumps(raw_record(i)) for i in (7,'7'))+'\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_export(src,out)
            self.assertFalse(out.exists())

    def test_write_failure_does_not_publish_partial_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'raw.jsonl';out=p/'features'
            src.write_text(json.dumps(raw_record())+'\n');original=Path.write_text
            def failing_write(path,*args,**kwargs):
                if path.name=='consensus_features.json':raise OSError('simulated disk write failure')
                return original(path,*args,**kwargs)
            with patch.object(Path,'write_text',failing_write),self.assertRaisesRegex(OSError,'simulated'):
                self.run_export(src,out)
            self.assertFalse(out.exists());self.assertFalse(list(p.glob('.features.tmp-*')))
            self.run_export(src,out);self.assertTrue((out/'feature_contract.json').is_file())

    def test_integer_manifest_maps_to_exported_string_id(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);src=p/'raw.jsonl';feature=p/'features'
            src.write_text(json.dumps(raw_record(7))+'\n');self.run_export(src,feature)
            manifest=p/'images.json';manifest.write_text(json.dumps([dict(id=7,image='7.jpg')]))
            args=['prepare','--data',str(feature),'--models',str(ROOT/'models'),
                  '--output',str(p/'pairs'),'--expected-images','1','--image-manifest',str(manifest)]
            with patch.object(sys,'argv',args),contextlib.redirect_stdout(io.StringIO()):prepare_pairs.main()
            self.assertTrue((p/'pairs/provenance.json').exists())

if __name__=='__main__':unittest.main()
