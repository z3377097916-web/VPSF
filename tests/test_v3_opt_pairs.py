"""单元测试：验证 E3_V3_OPT 非截断零泄漏准入、纯金牌锁定、双异构负例配对与每图等权归一化契约。"""
import unittest

from preference.purity import evaluate
from preference.selectors import normalized_weights
from scripts.v3_build_opt_pairs import inspect_unchopped_candidate
from v2.v3p_candidate import compile_strict_ledger


def _make_fact(cat="object", pol="positive", subj="dog", quote="a brown dog", statuses=("supported", "supported"), p=0.88):
    return {
        "category": cat,
        "polarity": pol,
        "modality": "definite",
        "subject": subj,
        "statement": f"There is {quote}.",
        "quote": quote,
        "quotes": [{"quote": quote}],
        "entities": [{"head": subj, "key": subj}],
        "counts": [3, 0, 0],
        "fresh_counts": [2, 0, 0],
        "checks": [
            {"method": "yesno", "status": statuses[0], "p_true": p, "valid": True, "low_mass": False},
            {"method": "yesno", "status": statuses[1], "p_true": p, "valid": True, "low_mass": False},
        ],
    }


class TestV3OptPairs(unittest.TestCase):
    def test_unchopped_guard_rejects_heavy_cut_or_and_zero_sup(self):
        facts = {
            "f1": _make_fact("object", "positive", "dog", "a brown dog", ("supported", "supported"), 0.91),
            "f2": _make_fact("object", "positive", "frisbee", "a red frisbee", ("supported", "supported"), 0.89),
            "f_zero": _make_fact("object", "positive", "umbrella", "an umbrella", ("unclear", "unclear"), 0.45),
        }
        comp = compile_strict_ledger(facts, {})
        good_text = (
            "A brown dog runs across a grassy park chasing a red frisbee on a sunny afternoon. "
            "Tall green trees line the background behind the open lawn while bright sunlight illuminates "
            "the wide field and casts soft natural shadows across the lush grass around the playful canine "
            "during a calm outdoor recreational scene."
        )
        w = len(good_text.split())
        self.assertGreaterEqual(w, 48)
        cand_ok = {"id": "img1:s0:0", "text": good_text, "words": w, "errors": [], "claim_ids": ["f1", "f2"]}
        meta = inspect_unchopped_candidate(cand_ok, orig_w=w + 4, facts=facts, gate={"passed": True}, compiled_ledger=comp)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["n_1sup"], 0)
        self.assertEqual(meta["cnt_1sup"], 0)
        self.assertTrue(meta["is_s0"])

        # 过短描述（words < 48）必须被拦截
        short_cand = dict(cand_ok, words=40)
        self.assertIsNone(
            inspect_unchopped_candidate(short_cand, orig_w=42, facts=facts, gate={"passed": True}, compiled_ledger=comp)
        )

        # 重度裁剪（cut = 24 > 18）必须被硬拦截，防止早停 EOS 偏置
        self.assertIsNone(
            inspect_unchopped_candidate(cand_ok, orig_w=w + 24, facts=facts, gate={"passed": True}, compiled_ledger=comp)
        )

        # 表面跨度存在跨候选 0_sup 具象名词（umbrella）时即使未进当前 claim_ids 也必须被拦截
        leak_text = good_text + " Near the bench stands an umbrella."
        cand_leak = {"id": "img1:s0:1", "text": leak_text, "words": len(leak_text.split()), "errors": [], "claim_ids": ["f1", "f2"]}
        self.assertIsNone(
            inspect_unchopped_candidate(cand_leak, orig_w=len(leak_text.split()), facts=facts, gate={"passed": True}, compiled_ledger=comp)
        )

        # 含 'or' 猜测短语必须被硬拦截
        or_text = good_text + " A bench or chair sits nearby."
        cand_or = {"id": "img1:s0:2", "text": or_text, "words": len(or_text.split()), "errors": [], "claim_ids": ["f1", "f2"]}
        self.assertIsNone(
            inspect_unchopped_candidate(cand_or, orig_w=len(or_text.split()), facts=facts, gate={"passed": True}, compiled_ledger=comp)
        )

    def test_purity_allows_natural_s0_against_bloated_multi_error_reject_but_blocks_severe_mix_inversion(self):
        # 76w 自然 s0 正例对比 106w 多错 s2 负例（delta_mix = -0.04, delta_qv = -0.09）应合法通过
        ok = evaluate(
            {"p": 0.94, "coverage": {"f": 0.26, "s": 0.26, "qv": 0.14}},
            {"p": 0.62, "coverage": {"f": 0.30, "s": 0.30, "qv": 0.23}},
        )
        self.assertTrue(ok["passed"])

        # 极端共识覆盖倒挂（delta_mix = -0.10 < -0.085）依然被拦截
        bad = evaluate(
            {"p": 0.94, "coverage": {"f": 0.20, "s": 0.20, "qv": 0.22}},
            {"p": 0.62, "coverage": {"f": 0.30, "s": 0.30, "qv": 0.20}},
        )
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["reason"], "severe_mix_coverage_inversion")

    def test_per_image_equal_weight_with_dual_distinct_negatives(self):
        rows = [
            {"image_id": "1", "chosen": "c1", "rejected": "r1"},
            {"image_id": "1", "chosen": "c1", "rejected": "r2"},
            {"image_id": "2", "chosen": "c2", "rejected": "r3"},
        ]
        weighted = normalized_weights(rows)
        w1 = sum(r["pair_weight"] for r in weighted if r["image_id"] == "1")
        w2 = sum(r["pair_weight"] for r in weighted if r["image_id"] == "2")
        self.assertAlmostEqual(w1, 1.5, places=12)
        self.assertAlmostEqual(w2, 1.5, places=12)


if __name__ == "__main__":
    unittest.main()
