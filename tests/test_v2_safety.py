"""Regression tests for evidence inflation, text safety and source replay."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'generation')]
from rlsfv.scoring import branch_provenance,weights
from v2.or_choice import decision,apply
from v2.storage import ImageStore,read


class V2Safety(unittest.TestCase):
    def test_repairs_are_not_fresh_votes(self):
        candidates=[]
        for stage,n in enumerate([10,7,5]):
            for i in range(n):
                candidates.append(dict(id=f'{stage}:{i}',stage=stage,branch='fresh' if stage==0 or i==n-1 else 'guided',claim_ids=['old'] if stage==0 and i==0 else []))
        facts={'old':dict(counts=[1,0,0]),'new':dict(counts=[0,0,0])}
        ledger=branch_provenance(candidates,facts)
        self.assertEqual(ledger['old']['fresh_counts'],[1,0,0]);self.assertEqual(ledger['new']['fresh_counts'],[0,0,0])
        repaired=copy.deepcopy(candidates)
        for c in repaired:c['claim_ids'].append('new')
        with self.assertRaisesRegex(ValueError,'counts disagree'):
            branch_provenance(repaired,facts)

    def test_or_abstains_on_low_mass_and_tie(self):
        check=lambda p,low=False:dict(p_yes=p,low_mass=low)
        self.assertIsNone(decision({'checks':[[check(.8),check(.8)],[check(.8),check(.8)]]},'g_argmax'))
        self.assertIsNone(decision({'checks':[[check(.9,True),check(.9)],[check(.1),check(.1)]]},'g_argmax'))
        self.assertEqual(decision({'checks':[[check(.9),check(.9)],[check(.1),check(.1)]]},'g_argmax'),'first')

    def test_or_does_not_strand_comparative_referent(self):
        from v2.repair import safe_or_application
        s='It might be a collision or a similar incident.'
        g=dict(a='a collision',b='a similar incident',first=[23,44],last=[12,26],
               checks=[[dict(p_yes=.7,low_mass=False)]*2,[dict(p_yes=.9,low_mass=False)]*2])
        result,choices=safe_or_application(s,[g])
        self.assertEqual(result,s);self.assertEqual(choices,[None])
        self.assertEqual(g['abstention_reason'],'winner_depends_on_removed_alternative')

    def test_or_deletion_does_not_add_space_before_comma(self):
        s='It is a rodent or cat, sitting on a desk.'
        a=s.index('or');b=s.index(',')
        g=dict(first=[a,b],last=[s.index('a rodent'),s.index('cat')],
               checks=[[dict(p_yes=.9,low_mass=False)]*2,[dict(p_yes=.1,low_mass=False)]*2])
        text,_=apply(s,[g],'g_argmax')
        self.assertEqual(text,'It is a rodent, sitting on a desk.')

    def test_cache_identity_prevents_stale_scores(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json.gz';store=ImageStore(p,{'model':'base','source':'a'})
            store.put({'text':'edited','prompt':'p'}, {'p':.9});store.flush()
            self.assertIsNone(store.peek({'text':'original','prompt':'p'}))
            self.assertEqual(ImageStore(p,store.identity).peek({'text':'edited','prompt':'p'}),{'p':.9})
            with self.assertRaisesRegex(ValueError,'identity changed'):ImageStore(p,{'model':'base','source':'b'})

    def test_overlapping_nominal_parses_abstain_without_blocking_other_groups(self):
        import spacy
        from v2.or_syntax import extract
        from v2.repair import safe_or_application
        nlp=spacy.load('en_core_web_sm')
        s=('The image features a person sitting at a dining table with various dishes and utensils in front of them. '
           'There are two main plates containing food, including what appears to be a burrito and a salad. '
           'A basket of bread or tortilla chips is also present on the table, adding to the variety of food items. \n\n'
           'A person is visible, sitting at the table and enjoying the meal. To complete the table setting, '
           'there are multiple cups and wine glasses placed around the table. One of the wine glasses is near '
           'the bottom of the table, while the others are positioned in different parts of the scene. '
           'A potted plant is located in the background, adding a touch of greenery to the setting. '
           'A chair can also be spotted near the left side of the table. A dog or cat is nearby.')
        groups=extract(nlp,s)
        self.assertEqual(len(groups),3)  # Reproduces the actual small-parser ambiguity.
        for g in groups:g['checks']=[[dict(p_yes=.9,low_mass=False)]*2,[dict(p_yes=.1,low_mass=False)]*2]
        repaired,choices=safe_or_application(s,groups)
        self.assertEqual(repaired,s.replace('A dog or cat is nearby.','A dog is nearby.'))
        self.assertEqual(choices,[None,None,'first'])
        self.assertTrue(all(g['abstention_reason']=='ambiguous_overlapping_alternatives' for g in groups[:2]))
        reversed_result,_=safe_or_application(s,list(reversed(groups)))
        self.assertEqual(repaired,reversed_result)

    def test_nominal_syntax_and_tail_boundaries(self):
        import spacy
        from v2.or_syntax import extract
        from v2.adjunct import edit
        from v2.cover_budget import edit as budget
        nlp=spacy.load('en_core_web_sm')
        self.assertEqual(extract(nlp,'There is no dog or cat in this room.'),[])
        s='A man is sitting on a chair, possibly using a phone or tablet. A dog sits nearby.'
        new,_=edit(nlp,s,'or_tail')
        self.assertEqual(new,'A man is sitting on a chair. A dog sits nearby.')
        s='A man is sitting on a chair, possibly watching a dog, a cat, and a horse.'
        new,_=budget(nlp(s),s,'spec_novel2');self.assertEqual(new,s)

if __name__=='__main__':unittest.main()
