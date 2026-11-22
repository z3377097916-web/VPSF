import copy,json,sys,tempfile,unittest,subprocess,os
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from scripts import parallel_pipeline as parallel
from scripts.run_training import validate_pair_source,validate_rows
from test_fix2 import raw_record
from rlsfv.scoring import score
from preference.source_contract import fingerprint

class ShardTests(unittest.TestCase):
    def test_10000_split_is_disjoint_balanced_and_keeps_ids(self):
        rows=[dict(id=str(i),image=str(i)) for i in range(10000)]
        shards=parallel.split_records(rows)
        self.assertEqual([len(s) for s in shards],[2500]*4)
        self.assertEqual({r['id'] for s in shards for r in s},{r['id'] for r in rows})

    def test_duplicate_normalized_id_rejected(self):
        with self.assertRaises(ValueError):parallel.split_records([{'id':i} for i in (1,'1',2,3)])

    def test_global_weights_recomputed(self):
        def r(i,c):return dict(image_id=i,prompt='P',chosen=c,rejected='R',pair_weight=1)
        merged=parallel.combine([[r('a','c'),r('a','d')],[r('b','c')],[],[]],[['a'],['b'],['c'],['d']],['a','b','c','d'])
        self.assertEqual([x['pair_weight'] for x in merged],[.75,.75,1.5])

    def test_missing_overlap_and_foreign_images_rejected(self):
        with self.assertRaises(ValueError):parallel.combine([[],[]],[['a'],['a']],['a'])
        with self.assertRaises(ValueError):parallel.combine([[]],[['a']],['a','b'])
        with self.assertRaises(ValueError):parallel.combine([[dict(image_id='b')]],[['a']],['a'])

    def test_four_workers_export_prepare_merge_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);cfg=json.loads((ROOT/'configs/run.json').read_text());cfg.update(output_root=str(p/'work'),expected_source_images=4)
            records=[]
            for i in range(4):
                image=p/f'{i}.jpg';image.write_bytes(f'fake image {i}'.encode());records.append(dict(id=str(i),image=str(image)))
            manifest=p/'manifest.json';manifest.write_text(json.dumps(records))
            plan=parallel.plan_run(cfg,ROOT/'configs/generation.json',manifest);parallel.install_plan(plan)
            original=subprocess.run
            def fake_generate(args,**kwargs):
                if '-m' not in args or 'rlsfv' not in args:return original(args,**kwargs)
                folder=Path(args[args.index('--output')+1]);mp=Path(args[args.index('--manifest')+1]);rs=json.loads(mp.read_text())
                folder.mkdir(parents=True,exist_ok=True)
                identity=None
                for record in rs:
                    d=raw_record();d['record']=record;d['image_sha256']=plan['image_hashes'][record['id']]
                    d['generation_identity']['config']=plan['generation'];identity=d['generation_identity']
                    d['facts']['dog']['checks']=[{'status':'supported'}]*2
                    d['ranking']=score(d['candidates'],d['facts'],plan['generation']['scoring'],42)
                    out=folder/'images'/record['id'];out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(d))
                (folder/'run.json').write_text(json.dumps(dict(identity,records=rs)))
                (folder/'status.json').write_text(json.dumps(dict(complete=True)))
                return subprocess.CompletedProcess(args,0)
            with patch.object(parallel.subprocess,'run',side_effect=fake_generate):
                for i in range(4):parallel.worker(plan,i)
            out=parallel.merge(plan);prov=parallel.read(out/'provenance.json');rows=parallel.read(out/'E3_combined.json')
            validate_pair_source(cfg,prov);validate_rows(rows,prov)
            self.assertEqual(prov['source_image_count'],4);self.assertEqual(len({r['image_id'] for r in rows}),4)
            sums={i:sum(r['pair_weight'] for r in rows if r['image_id']==i) for i in map(str,range(4))}
            self.assertEqual(len(set(sums.values())),1)
            self.assertEqual(out,parallel.merge(plan))
            with patch.object(parallel,'idle_cards'),patch.object(parallel.subprocess,'run',side_effect=AssertionError('completed shard must not restart')):
                parallel.run_workers(plan,p/'unused.json')
            done=Path(plan['shards'][0]['directory'])/'complete.json';state=parallel.read(done);state['plan_hash']='changed';done.write_text(json.dumps(state))
            with self.assertRaises(ValueError):parallel.merge(plan)

if __name__=='__main__':unittest.main()
