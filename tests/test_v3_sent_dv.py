"""E3-V3Z 句级双核查零泄漏外科裁剪模块（v2.sent_dv）单元测试。"""
import sys
import unittest
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "generation")]

from v2.sent_dv import check_fact_strict_dv, has_broken_backref, sent_nouns


class TestV3SentDV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def test_single_vote_concrete_rejected_while_double_vote_passes(self):
        # 单票 ['supported', 'unclear'] 具象物体必须被拦截（防止单点共现偏置泄漏拉高篇级 Hal）
        f_single = {
            "category": "object",
            "polarity": "positive",
            "modality": "definite",
            "subject": "handbag",
            "entities": [{"head": "handbag"}],
            "counts": [3, 1, 0],
            "checks": [
                {"status": "supported", "p_true": 0.78, "low_mass": False, "valid": True},
                {"status": "unclear", "p_true": 0.56, "low_mass": False, "valid": True},
            ],
        }
        self.assertFalse(check_fact_strict_dv(f_single))

        # 双票 ['supported', 'supported'] 高共识真物体通过
        f_double = {
            "category": "object",
            "polarity": "positive",
            "modality": "definite",
            "subject": "bench",
            "entities": [{"head": "bench"}],
            "counts": [5, 2, 1],
            "checks": [
                {"status": "supported", "p_true": 0.88, "low_mass": False, "valid": True},
                {"status": "supported", "p_true": 0.85, "low_mass": False, "valid": True},
            ],
        }
        self.assertTrue(check_fact_strict_dv(f_double))

    def test_singleton_low_confidence_mirage_blocked(self):
        # 仅在 1 条候选出现且双核查仅擦边 0.66 的孤立幻觉物体被拦截
        f_mirage = {
            "category": "object",
            "polarity": "positive",
            "modality": "definite",
            "subject": "spoon",
            "entities": [{"head": "spoon"}],
            "counts": [0, 1, 0],
            "checks": [
                {"status": "supported", "p_true": 0.66, "low_mass": False, "valid": True},
                {"status": "supported", "p_true": 0.67, "low_mass": False, "valid": True},
            ],
        }
        self.assertFalse(check_fact_strict_dv(f_mirage))

        # 引导轮发现但双核查高置信（>=0.80）的真实细节正常保留
        f_clear_guided = dict(
            f_mirage,
            checks=[
                {"status": "supported", "p_true": 0.84, "low_mass": False, "valid": True},
                {"status": "supported", "p_true": 0.81, "low_mass": False, "valid": True},
            ],
        )
        self.assertTrue(check_fact_strict_dv(f_clear_guided))

    def test_broken_definite_backref_and_dangling_pronoun_blocked(self):
        doc = self.nlp(
            "A woman stands in a park. "
            "Near her is a wooden bench with a purse or backpack. "
            "On the bench sits a small dog. "
            "Large green trees line the walkway."
        )
        sents = list(doc.sents)
        kept = sent_nouns(sents[0])
        dropped = sent_nouns(sents[1])
        # 第 2 句引用了仅在被删第 1 句中引入的 'the bench'，必须被拦截
        self.assertTrue(has_broken_backref(sents[2], True, kept, dropped))
        # 第 3 句无悬垂指代且未引用被删句专属实体，可安全保留
        self.assertFalse(has_broken_backref(sents[3], True, kept, dropped | sent_nouns(sents[2])))


if __name__ == "__main__":
    unittest.main()
