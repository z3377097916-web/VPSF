"""单元测试：验证 E3_V3_OPT3 全槽位词干对齐、Foil自校准账本、推测名词拦截与句尾空话剥离契约。"""
import unittest

from scripts.v3_build_opt3_pairs import (
    _trim_closing_fluff,
    canon_noun_phrase,
    check_fact_opt3,
    compile_calibrated_ledger_v3,
    extract_all_concrete_heads,
    inspect_opt3_candidate,
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


class TestV3Opt3Pairs(unittest.TestCase):
    def test_multi_slot_and_plural_lemma_extraction(self):
        ok, ch = canon_noun_phrase("wooden benches")
        self.assertTrue(ok)
        self.assertEqual(ch, "bench")

        f_rel = _make_fact(
            cat="spatial",
            subj="man",
            quote="man sitting on benches",
            statuses=("supported", "unclear"),
            entities=[{"head": "man", "key": "man"}, {"anchor": "benches", "key": "benches"}],
            align_obj="benches",
        )
        heads = extract_all_concrete_heads(f_rel)
        self.assertIn("man", heads)
        self.assertIn("bench", heads)
        # 若 bench 已在准入实体集中，spatial 的 1_sup/0_sup 修饰不应被误杀；若 bench 未准入，必须硬拦截！
        self.assertTrue(check_fact_opt3(f_rel, allowed_heads={"man", "bench"}))
        self.assertFalse(check_fact_opt3(f_rel, allowed_heads={"man"}))

    def test_identity_predicate_nominal_and_repaired_zero_counts_rescue(self):
        # identity 表语名词 surfboard 若未核准，必须拦截
        f_id = _make_fact(
            cat="identity",
            subj="object",
            quote="looks like a surfboard",
            statuses=("supported", "unclear"),
            entities=[{"head": "object", "key": "object"}],
            align_obj="surfboard",
        )
        self.assertFalse(check_fact_opt3(f_id, allowed_heads={"person"}))
        self.assertTrue(check_fact_opt3(f_id, allowed_heads={"surfboard"}))

        # 修复事实 counts=[0,0,0] 应从全图同词头原始事实继承票数，不被误判为孤立单样本
        facts = {
            "f_orig": _make_fact("object", "positive", "definite", "dog", "a dog", ("supported", "supported"), 0.69, counts=(4, 0, 0)),
            "f_rep": _make_fact(
                "object",
                "positive",
                "definite",
                "dog",
                "the brown dog",
                ("supported", "supported"),
                0.69,
                counts=(0, 0, 0),
                fresh_counts=(0, 0, 0),
                orig_absent=True,
            ),
            # 孤立引导单样本（counts=[1,0,0], fresh=[0,0,0], p=0.69 < 0.72）在高 foil_fsr 下必须被降级拦截
            "f_iso": _make_fact(
                "object",
                "positive",
                "definite",
                "bench",
                "a bench",
                ("supported", "supported"),
                0.69,
                counts=(1, 0, 0),
                fresh_counts=(0, 0, 0),
            ),
        }
        _, gold_h, _ = compile_calibrated_ledger_v3(facts, {}, foil_fsr=0.30)
        self.assertIn("dog", gold_h)
        self.assertNotIn("bench", gold_h)

    def test_epistemic_noun_hedge_blocked_benign_mood_allowed_and_fluff_trimmed(self):
        facts = {
            "f1": _make_fact("object", "positive", "definite", "dog", "a brown dog", ("supported", "supported"), 0.92),
            "f2": _make_fact("object", "positive", "definite", "frisbee", "a red frisbee", ("supported", "supported"), 0.90),
            "f3": _make_fact("object", "positive", "definite", "bench", "a wooden bench", ("supported", "supported"), 0.89),
        }
        comp, gold_h, cons_h = compile_calibrated_ledger_v3(facts, {}, foil_fsr=0.10)
        aligner = _StubAligner({"a brown dog": "f1", "a red frisbee": "f2", "a wooden bench": "f3"})

        core = (
            "The image features a brown dog running energetically across an open grassy park while chasing a red frisbee on a sunny afternoon. "
            "Bright natural sunlight illuminates the wide green lawn around the playful canine as the animals appear happy together.\n\n"
            "In addition to the dog, a wooden bench sits quietly beside the paved walkway near the shade of the park."
        )
        with_fluff = core + " The overall atmosphere of the outdoor scene is serene, harmonious, and inviting."

        trimmed, n_cuts, _ = _trim_closing_fluff(with_fluff, comp, min_kept_words=58)
        self.assertEqual(n_cuts, 1)
        self.assertEqual(trimmed, core)
        self.assertIn("\n\n", trimmed)

        rc = {
            "id": "img1:s0:0",
            "text": with_fluff,
            "facts": [{"quote": "a brown dog"}, {"quote": "a red frisbee"}, {"quote": "a wooden bench"}],
            "claim_ids": ["f1", "f2", "f3"],
            "fuzzy": {"u": 0.18},
        }
        res = inspect_opt3_candidate(
            rc=rc,
            repaired_txt=with_fluff,
            orig_w=len(with_fluff.split()),
            ledger_facts=facts,
            gate_info={"passed": True},
            comp=comp,
            gold_heads=gold_h,
            cons_heads=cons_h,
            aligner=aligner,
            vocab=None,
        )
        self.assertIsNotNone(res)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1]["text"], core)

        # 若含猜名词推测短语（如 possibly / might be / looks like a），必须被硬拦截
        hedged = "It looks like a surfboard near the grass. " + core
        rc_hedged = dict(rc, text=hedged)
        self.assertIsNone(
            inspect_opt3_candidate(
                rc=rc_hedged,
                repaired_txt=hedged,
                orig_w=len(hedged.split()),
                ledger_facts=facts,
                gate_info={"passed": True},
                comp=comp,
                gold_heads=gold_h,
                cons_heads=cons_h,
                aligner=aligner,
                vocab=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
