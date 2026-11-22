"""E3-V3P 核心模块单元测试（双核查跨度保护、洁净引导清单、词法多样性饱和、子句外科剥离与姿态动词守卫）。"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from rlsfv.questions import guided_compact, verified_facts
from v2.ledger_guard import build_ledger_span_guard
from v2.v3p_candidate import (
    check_fact_strict_v3p,
    compile_strict_ledger,
    effective_dv_diversity_score,
    try_spacy_tail_trim,
)


class TestV3PCandidate(unittest.TestCase):

    def test_ledger_span_guard_requires_double_support(self):
        facts = {
            "f_ds": {
                "category": "object",
                "polarity": "positive",
                "subject": "bench",
                "entities": [{"head": "bench"}],
                "checks": [
                    {"status": "supported", "low_mass": False},
                    {"status": "supported", "low_mass": False},
                ],
            },
            "f_ss": {
                "category": "object",
                "polarity": "positive",
                "subject": "handbag",
                "entities": [{"head": "handbag"}],
                "checks": [
                    {"status": "supported", "low_mass": False},
                    {"status": "unclear", "low_mass": False},
                ],
            },
        }
        guard = build_ledger_span_guard(["f_ds", "f_ss"], facts)
        self.assertIn("bench", guard["supported_heads"])
        self.assertNotIn("handbag", guard["supported_heads"])
        self.assertIn("handbag", guard["unsafe_heads"])

    def test_diversity_score_merges_synonyms_and_caps_person_cluster(self):
        # 同一人类簇换词（person, man, individual）最多只计 1.5，同义词 bike/bicycle 仅计 1
        heads_person_heavy = {"person", "man", "individual", "bike", "bicycle", "bench"}
        self.assertAlmostEqual(effective_dv_diversity_score(heads_person_heavy), 2.0 + 1.5, places=5)

        # 多样化真实物体（man + bicycle + bench + cup + backpack）得分显著更高
        heads_diverse = {"man", "bicycle", "bench", "cup", "backpack"}
        self.assertAlmostEqual(effective_dv_diversity_score(heads_diverse), 4.0 + 1.0, places=5)
        self.assertGreater(effective_dv_diversity_score(heads_diverse), effective_dv_diversity_score(heads_person_heavy))

    def test_strict_v3p_fact_requires_double_supported_count_and_object(self):
        cnt_1sup = {
            "category": "count",
            "polarity": "positive",
            "modality": "definite",
            "subject": "chair",
            "object": "3",
            "checks": [
                {"status": "supported", "p_true": 0.78, "low_mass": False, "valid": True},
                {"status": "unclear", "p_true": 0.55, "low_mass": False, "valid": True},
            ],
        }
        self.assertFalse(check_fact_strict_v3p(cnt_1sup))

        cnt_2sup = dict(
            cnt_1sup,
            checks=[
                {"status": "supported", "p_true": 0.85, "low_mass": False, "valid": True},
                {"status": "supported", "p_true": 0.82, "low_mass": False, "valid": True},
            ],
        )
        self.assertTrue(check_fact_strict_v3p(cnt_2sup))

    def test_spacy_tail_trim_saves_comma_adjunct_but_blocks_sitting_complement(self):
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
        facts = {
            "f_table": {
                "category": "object",
                "polarity": "positive",
                "subject": "table",
                "quote": "wooden table",
                "counts": [4, 2, 1],
                "entities": [{"head": "table"}],
                "checks": [
                    {"status": "supported", "p_true": 0.90, "low_mass": False, "valid": True},
                    {"status": "supported", "p_true": 0.88, "low_mass": False, "valid": True},
                ],
            },
            "f_whisk": {
                "category": "object",
                "polarity": "positive",
                "subject": "whisk",
                "quote": "a whisk",
                "counts": [3, 1, 1],
                "entities": [{"head": "whisk"}],
                "checks": [
                    {"status": "supported", "p_true": 0.88, "low_mass": False, "valid": True},
                    {"status": "supported", "p_true": 0.86, "low_mass": False, "valid": True},
                ],
            },
            "f_bag_1sup": {
                "category": "object",
                "polarity": "positive",
                "subject": "handbag",
                "quote": "accompanied by a small handbag",
                "counts": [1, 1, 0],
                "entities": [{"head": "handbag"}],
                "checks": [
                    {"status": "supported", "p_true": 0.72, "low_mass": False, "valid": True},
                    {"status": "unclear", "p_true": 0.54, "low_mass": False, "valid": True},
                ],
            },
            "f_bowl_1sup": {
                "category": "object",
                "polarity": "positive",
                "subject": "bowl",
                "quote": "in front of a bowl",
                "counts": [1, 0, 0],
                "entities": [{"head": "bowl"}],
                "checks": [
                    {"status": "supported", "p_true": 0.70, "low_mass": False, "valid": True},
                    {"status": "unclear", "p_true": 0.51, "low_mass": False, "valid": True},
                ],
            },
        }
        comp = compile_strict_ledger(facts, None)

        # 合法逗号伴随短语剥离：挽救句首主干
        s_good = "A woman sits at a large wooden table in the room, accompanied by a small handbag."
        doc_good = nlp(s_good)
        res_good = try_spacy_tail_trim(doc_good, list(doc_good.sents)[0], [facts["f_bag_1sup"]], comp)
        self.assertIsNotNone(res_good)
        self.assertEqual(res_good[0], "A woman sits at a large wooden table in the room.")

        # 姿态分词补语保护：严禁将 'with a whisk sitting in front of a bowl.' 截断为残句 'with a whisk sitting.'
        s_bad = "A counter is shown with a whisk sitting in front of a bowl."
        doc_bad = nlp(s_bad)
        res_bad = try_spacy_tail_trim(doc_bad, list(doc_bad.sents)[0], [facts["f_bowl_1sup"]], comp)
        self.assertIsNone(res_bad)

    def test_v3p_clean_guided_prompt_and_singleton_filter(self):
        facts = {
            "ok": {
                "category": "object",
                "polarity": "positive",
                "modality": "definite",
                "counts": [3, 0, 0],
                "checks": [
                    {"status": "supported", "p_true": 0.88, "low_mass": False, "valid": True},
                    {"status": "supported", "p_true": 0.85, "low_mass": False, "valid": True},
                ],
            },
            "low_singleton": {
                "category": "object",
                "polarity": "positive",
                "modality": "definite",
                "counts": [1, 0, 0],
                "checks": [
                    {"status": "supported", "p_true": 0.66, "low_mass": False, "valid": True},
                    {"status": "supported", "p_true": 0.67, "low_mass": False, "valid": True},
                ],
            },
        }
        vf = verified_facts(facts, min_support=1.0, min_singleton_p=0.68)
        self.assertIn("ok", vf)
        self.assertNotIn("low_singleton", vf)

        qs = [{"render": "existence", "text": "bench"}, {"render": "open", "text": "Look carefully."}]
        prompt = guided_compact(qs, style="v3p_clean", relation_module=True)
        self.assertNotIn("absent, or unclear", prompt)
        self.assertIn("never use 'or'", prompt)


if __name__ == "__main__":
    unittest.main()
