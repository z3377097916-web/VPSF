"""单元测试：验证 E3_V3_OPT4 97%+ 纯双支持账本、修复单样本防漏、计数分级与自然段落保留契约。"""
import unittest

from scripts.v3_build_opt4_pairs import (
    check_fact_opt4,
    compile_opt4_ledger,
    inspect_opt4_candidate,
)


def _make_fact(
    cat="object",
    pol="positive",
    mod="definite",
    subj="dog",
    quote="a brown dog",
    statuses=("supported", "supported"),
    p=0.90,
    entities=None,
    align_obj="",
    counts=(3, 0, 0),
    fresh_counts=(2, 0, 0),
    orig_absent=False,
):
    f = {
        "category": cat,
        "polarity": pol,
        "modality": mod,
        "subject": subj,
        "statement": f"There is {quote}.",
        "quote": quote,
        "quotes": [{"quote": quote}],
        "entities": entities if entities is not None else [{"head": subj, "key": subj}],
        "counts": list(counts),
        "fresh_counts": list(fresh_counts),
        "original_evidence_absent": orig_absent,
        "checks": [
            {"method": "yesno", "status": statuses[0], "p_true": p, "valid": True, "low_mass": False},
            {"method": "yesno", "status": statuses[1], "status_raw": statuses[1], "p_true": p, "valid": True, "low_mass": False},
        ],
    }
    if align_obj:
        f["align"] = {"object": align_obj}
        if cat == "identity":
            f["object"] = align_obj
    return f


class _StubAligner:
    def __init__(self, key_map):
        self._map = key_map

    def canonical(self, lf, cid):
        return self._map[lf["quote"]]


class TestV3Opt4Pairs(unittest.TestCase):
    def test_opt4_ledger_repaired_consensus_vs_repaired_singleton(self):
        facts = {
            "f_orig": _make_fact(
                "object", "positive", "definite", "dog", "a dog", ("supported", "supported"), 0.70, counts=(4, 0, 0)
            ),
            # 同实体跨候选有 4 票支撑的微修事实应正常进入 gold_dv_heads
            "f_rep_ok": _make_fact(
                "object",
                "positive",
                "definite",
                "dog",
                "the brown dog",
                ("supported", "supported"),
                0.70,
                counts=(0, 0, 0),
                fresh_counts=(0, 0, 0),
                orig_absent=True,
            ),
            # 全图无原始票数支撑的孤立微修新名词（counts=[0,0,0], p=0.70 < 0.73）必须被降级拦截
            "f_rep_iso": _make_fact(
                "object",
                "positive",
                "definite",
                "umbrella",
                "an umbrella",
                ("supported", "supported"),
                0.70,
                counts=(0, 0, 0),
                fresh_counts=(0, 0, 0),
                orig_absent=True,
            ),
            # 高共识单支持实体只允许进入 cons_dv_heads，绝不污染 gold_dv_heads
            "f_cons1": _make_fact(
                "object",
                "positive",
                "definite",
                "chair",
                "a chair",
                ("supported", "unclear"),
                0.72,
                counts=(5, 0, 0),
                fresh_counts=(3, 0, 0),
            ),
        }
        _, gold_h, cons_h = compile_opt4_ledger(facts, {}, foil_fsr=0.28)
        self.assertIn("dog", gold_h)
        self.assertNotIn("umbrella", gold_h)
        self.assertNotIn("chair", gold_h)
        self.assertIn("chair", cons_h)

    def test_check_fact_opt4_count_and_multislot_rules(self):
        f_cnt1 = _make_fact("count", "positive", "definite", "dog", "two dogs", ("supported", "unclear"), 0.70)
        self.assertFalse(check_fact_opt4(f_cnt1, {"dog"}, allow_1sup_cnt=False))
        self.assertTrue(check_fact_opt4(f_cnt1, {"dog"}, allow_1sup_cnt=True))

        f_spat = _make_fact(
            "spatial",
            "positive",
            "definite",
            "dog",
            "dog beside wooden benches",
            ("supported", "unclear"),
            0.75,
            entities=[{"head": "dog", "key": "dog"}, {"anchor": "benches", "key": "benches"}],
            align_obj="benches",
        )
        self.assertFalse(check_fact_opt4(f_spat, {"dog"}, allow_1sup_cnt=True))
        self.assertTrue(check_fact_opt4(f_spat, {"dog", "bench"}, allow_1sup_cnt=True))

    def test_inspect_opt4_candidate_preserves_natural_ending_and_isolates_gold(self):
        facts = {
            "f1": _make_fact("object", "positive", "definite", "dog", "a brown dog", ("supported", "supported"), 0.92),
            "f2": _make_fact("object", "positive", "definite", "frisbee", "a red frisbee", ("supported", "supported"), 0.90),
            "f3": _make_fact(
                "object",
                "positive",
                "definite",
                "bench",
                "a wooden bench",
                ("supported", "unclear"),
                0.74,
                counts=(5, 0, 0),
                fresh_counts=(3, 0, 0),
            ),
        }
        comp, gold_h, cons_h = compile_opt4_ledger(facts, {}, foil_fsr=0.10)
        aligner = _StubAligner({"a brown dog": "f1", "a red frisbee": "f2", "a wooden bench": "f3"})

        natural_1p = (
            "The image features a brown dog running energetically across an open grassy park while chasing a red frisbee on a sunny afternoon. "
            "Bright natural sunlight illuminates the wide green lawn around the playful canine throughout the scene. "
            "The overall atmosphere of the outdoor park is calm, peaceful, and inviting."
        )
        rc_gold = {
            "id": "img1:s0:0",
            "text": natural_1p,
            "facts": [{"quote": "a brown dog"}, {"quote": "a red frisbee"}],
            "claim_ids": ["f1", "f2"],
            "fuzzy": {"u": 0.16},
        }
        res_gold = inspect_opt4_candidate(
            rc=rc_gold,
            repaired_txt=natural_1p,
            orig_w=len(natural_1p.split()),
            ledger_facts=facts,
            gate_info={"passed": True},
            comp=comp,
            gold_heads=gold_h,
            cons_heads=cons_h,
            aligner=aligner,
            vocab=None,
            allow_edit=False,
        )
        self.assertIsNotNone(res_gold)
        self.assertEqual(res_gold[0], 1)
        # OPT4 不强行剥离自然单段句尾总结，杜绝强制 100% 双段长度失真
        self.assertEqual(res_gold[1]["text"], natural_1p)
        self.assertEqual(res_gold[2]["n_cons1"], 0)

        # 含 cons1 实体（bench）的候选绝不能进入 Tier 1/2 纯金池，只能落入 Tier 3
        cons_txt = natural_1p + " A wooden bench rests beside the path."
        rc_cons = {
            "id": "img1:s0:1",
            "text": cons_txt,
            "facts": [{"quote": "a brown dog"}, {"quote": "a red frisbee"}, {"quote": "a wooden bench"}],
            "claim_ids": ["f1", "f2", "f3"],
            "fuzzy": {"u": 0.19},
        }
        res_cons = inspect_opt4_candidate(
            rc=rc_cons,
            repaired_txt=cons_txt,
            orig_w=len(cons_txt.split()),
            ledger_facts=facts,
            gate_info={"passed": True},
            comp=comp,
            gold_heads=gold_h,
            cons_heads=cons_h,
            aligner=aligner,
            vocab=None,
            allow_edit=False,
        )
        self.assertIsNotNone(res_cons)
        self.assertEqual(res_cons[0], 3)
        self.assertEqual(res_cons[2]["n_cons1"], 1)


if __name__ == "__main__":
    unittest.main()
