"""E3-V3 核心模块单元测试（具象边缘预算、账本保护、多元 or 择优、双维纯净度、对称长度负例、只读缓存校验）。"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'generation')]

from v2.ledger_guard import is_concrete_entity, candidate_v3_diagnostics, build_ledger_span_guard
from v2.or_choice import multiway_decision, apply_multiway
from v2.storage import ImageStore
from preference.purity import evaluate
from preference.selectors import choose_v3, rejected_v3


class TestV3Upgrades(unittest.TestCase):
    def test_concrete_vs_abstract_entities(self):
        self.assertEqual(is_concrete_entity({'subject': 'dining table', 'entities': [{'head': 'dining table'}]}), (True, 'dining table'))
        self.assertEqual(is_concrete_entity({'subject': 'cell phone', 'entities': [{'head': 'cell phone'}]}), (True, 'cell phone'))
        self.assertEqual(is_concrete_entity({'subject': 'tie', 'entities': [{'head': 'tie'}]}), (True, 'tie'))
        self.assertEqual(is_concrete_entity({'subject': 'viewer', 'entities': [{'head': 'viewer'}]})[0], False)
        self.assertEqual(is_concrete_entity({'subject': 'combination', 'entities': [{'head': 'combination'}]})[0], False)
        self.assertEqual(is_concrete_entity({'subject': 'peaceful atmosphere', 'entities': [{'head': 'peaceful atmosphere'}]})[0], False)

    def test_edge_risk_budget_does_not_tax_double_supported_objects(self):
        facts = {}
        claim_ids = []
        # 构造 12 个双模板全支持的真物体（高 Cover 场景）
        for i in range(12):
            cid = f'obj_{i}'
            claim_ids.append(cid)
            facts[cid] = {
                'category': 'object', 'polarity': 'positive', 'modality': 'definite',
                'subject': f'chair_{i}', 'entities': [{'head': f'chair_{i}'}],
                'checks': [
                    {'status': 'supported', 'p_true': 0.88, 'low_mass': False, 'valid': True},
                    {'status': 'supported', 'p_true': 0.86, 'low_mass': False, 'valid': True},
                ],
            }
        # 再加 1 个单模板支持（A=0.76 supported, B=0.62 unclear, mean=0.69 -> r_edge=0.31 <= 0.45）的边缘真物体
        claim_ids.append('edge_bag')
        facts['edge_bag'] = {
            'category': 'object', 'polarity': 'positive', 'modality': 'definite',
            'subject': 'handbag', 'entities': [{'head': 'handbag'}],
            'checks': [
                {'status': 'supported', 'p_true': 0.76, 'low_mass': False, 'valid': True},
                {'status': 'unclear', 'p_true': 0.62, 'low_mass': False, 'valid': True},
            ],
        }
        c_feat = {'text': 'Twelve chairs and a handbag are in the room.', 'errors': [], 'claim_ids': claim_ids, 'coverage': {'f': 0.4, 's': 0.3, 'qv': 0.35}}
        diag = candidate_v3_diagnostics(c_feat, facts, gate={'passed': True, 'pending_checks': {}}, max_r_edge=0.45)
        self.assertTrue(diag['strict_no_or_safe'])
        self.assertAlmostEqual(diag['r_edge'], round(0.35 * 0.31, 6), places=4)
        self.assertEqual(diag['n_ds_obj'], 12)

        # 若混入双模板零支持的具象猜测（['unclear', 'unclear']），必须被严格拦截
        claim_ids2 = claim_ids + ['zero_sup_tv']
        facts['zero_sup_tv'] = {
            'category': 'object', 'polarity': 'positive', 'modality': 'uncertain',
            'subject': 'tv', 'entities': [{'head': 'tv'}],
            'checks': [
                {'status': 'unclear', 'p_true': 0.52, 'low_mass': False, 'valid': True},
                {'status': 'unclear', 'p_true': 0.49, 'low_mass': False, 'valid': True},
            ],
        }
        c_feat2 = dict(c_feat, claim_ids=claim_ids2)
        diag2 = candidate_v3_diagnostics(c_feat2, facts, gate={'passed': True, 'pending_checks': {}}, max_r_edge=0.45)
        self.assertFalse(diag2['base_safe'])
        self.assertEqual(diag2['zero_sup_concrete'], 1)

    def test_negated_absence_clause_blocked(self):
        c_feat = {'text': 'A quiet street with a bench, with no cars or people visible.', 'errors': [], 'claim_ids': [], 'coverage': {'f': 0.3, 's': 0.3, 'qv': 0.2}}
        diag = candidate_v3_diagnostics(c_feat, {}, gate={'passed': True, 'pending_checks': {}})
        self.assertTrue(diag['has_neg_clause'])
        self.assertFalse(diag['base_safe'])

    def test_multiway_or_resolution(self):
        text = 'The desk holds crafts, office tasks, or DIY projects near the lamp.'
        s = text.index('crafts')
        e = text.index('projects') + len('projects')
        mg = {
            'full_span': [s, e],
            'options': [
                {'text': 'crafts', 'replacement': 'crafts'},
                {'text': 'office tasks', 'replacement': 'office tasks'},
                {'text': 'DIY projects', 'replacement': 'DIY projects'},
            ],
            'checks': [
                [{'p_yes': 0.25, 'low_mass': False}, {'p_yes': 0.20, 'low_mass': False}],
                [{'p_yes': 0.88, 'low_mass': False}, {'p_yes': 0.84, 'low_mass': False}],
                [{'p_yes': 0.30, 'low_mass': False}, {'p_yes': 0.28, 'low_mass': False}],
            ],
        }
        self.assertEqual(multiway_decision(mg), 1)
        repaired, decs = apply_multiway(text, [mg])
        self.assertEqual(decs, [1])
        self.assertEqual(repaired, 'The desk holds office tasks near the lamp.')

    def test_purity_joint_coverage_and_backward_compat(self):
        # 向后兼容仅含 p 字段的校验调用
        self.assertTrue(evaluate({'p': 0.92}, {'p': 0.60})['passed'])
        # 拦截严重覆盖率倒挂
        bad_mix = evaluate({'p': 0.95, 'coverage': {'f': 0.20, 's': 0.20, 'qv': 0.25}},
                           {'p': 0.60, 'coverage': {'f': 0.30, 's': 0.30, 'qv': 0.20}})
        self.assertFalse(bad_mix['passed'])
        self.assertEqual(bad_mix['reason'], 'severe_mix_coverage_inversion')
        # 正常覆盖率优势通过
        ok = evaluate({'p': 0.95, 'coverage': {'f': 0.32, 's': 0.30, 'qv': 0.22}},
                      {'p': 0.65, 'coverage': {'f': 0.20, 's': 0.18, 'qv': 0.14}})
        self.assertTrue(ok['passed'])

    def test_storage_readonly_fallback_checks_identity(self):
        with tempfile.TemporaryDirectory() as d:
            v2_p = Path(d) / 'v2.json.gz'
            v3_p = Path(d) / 'v3.json.gz'
            s2 = ImageStore(v2_p, {'run': 'v2', 'image_sha256': 'img1', 'source_sha256': 'src1'})
            s2.put({'kind': 'fuzzy', 'text': 'abc'}, {'u': 0.12})
            s2.flush()
            # 同图同源允许安全命中
            s3_ok = ImageStore(v3_p, {'run': 'v3', 'image_sha256': 'img1', 'source_sha256': 'src1'}, fallback_paths=[v2_p])
            self.assertEqual(s3_ok.peek({'kind': 'fuzzy', 'text': 'abc'}), {'u': 0.12})
            # 源哈希不一致时严禁复用
            s3_diff = ImageStore(Path(d) / 'v3_b.json.gz', {'run': 'v3', 'image_sha256': 'img1', 'source_sha256': 'src_other'}, fallback_paths=[v2_p])
            self.assertIsNone(s3_diff.peek({'kind': 'fuzzy', 'text': 'abc'}))

    def test_adj_or_and_speculative_or_sentence_edit(self):
        import spacy
        from v2.repair import _adj_or_edit, _speculative_or_sentence_edit
        nlp = spacy.load('en_core_web_sm')
        s1 = 'The overall scene suggests a professional or formal event.'
        out1, sp1 = _adj_or_edit(nlp(s1), s1)
        self.assertEqual(out1, 'The overall scene suggests a professional event.')
        self.assertEqual(len(sp1), 1)

        s2 = (
            'The image features a cracked red fire hydrant on the sidewalk near the street. '
            'Two old television screens are resting on top of the broken hydrant in broad daylight. '
            'The scene appears as if it might have been the result of a vehicle collision or a similar incident.'
        )
        out2, sp2 = _speculative_or_sentence_edit(nlp(s2), s2)
        self.assertNotIn(' or ', out2.lower())
        self.assertTrue(out2.endswith('daylight.'))
        self.assertEqual(len(sp2), 1)


if __name__ == '__main__':
    unittest.main()
