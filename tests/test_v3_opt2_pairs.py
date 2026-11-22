"""单元测试：验证 E3_V3_OPT2 双段高密度纯金牌准入、被裁句尾正归属安全契约与 100% 正例去重契约。"""
import unittest

from scripts.v3_build_opt2_pairs import build_p_map, inspect_single_candidate_all_tiers
from v2.v3p_candidate import compile_strict_ledger


def _make_fact(cat="object", pol="positive", subj="dog", quote="a brown dog", statuses=("supported", "supported"), p=0.90, counts=(3, 0, 0)):
    return {
        "category": cat,
        "polarity": pol,
        "modality": "definite",
        "subject": subj,
        "statement": f"There is {quote}.",
        "quote": quote,
        "quotes": [{"quote": quote}],
        "entities": [{"head": subj, "key": subj}],
        "counts": list(counts),
        "fresh_counts": [2, 0, 0],
        "checks": [
            {"method": "yesno", "status": statuses[0], "p_true": p, "valid": True, "low_mass": False},
            {"method": "yesno", "status": statuses[1], "p_true": p, "valid": True, "low_mass": False},
        ],
    }


class _StubAligner:
    def __init__(self, key_map):
        self._map = key_map

    def canonical(self, lf, cid):
        return self._map[lf["quote"]]


class TestV3Opt2Pairs(unittest.TestCase):
    def test_gold_full_preserves_two_paragraphs_and_computes_p2_novel(self):
        facts = {
            "f1": _make_fact("object", "positive", "dog", "a brown dog", ("supported", "supported"), 0.92),
            "f2": _make_fact("object", "positive", "frisbee", "a red frisbee", ("supported", "supported"), 0.90),
            "f3": _make_fact("object", "positive", "bench", "a wooden bench", ("supported", "supported"), 0.89),
        }
        comp = compile_strict_ledger(facts, {})
        p_map = build_p_map(facts, {})
        aligner = _StubAligner({"a brown dog": "f1", "a red frisbee": "f2", "a wooden bench": "f3"})

        txt = (
            "The image features a brown dog running energetically across an open grassy park while chasing a red frisbee. "
            "Bright afternoon sunlight illuminates the lush green lawn around the playful canine as it leaps forward.\n\n"
            "In addition to the dog, a wooden bench sits quietly beside the paved walkway near the tall shade trees."
        )
        rc = {
            "id": "img1:s0:0",
            "text": txt,
            "facts": [{"quote": "a brown dog"}, {"quote": "a red frisbee"}, {"quote": "a wooden bench"}],
            "claim_ids": ["f1", "f2", "f3"],
            "fuzzy": {"u": 0.18},
        }
        res = inspect_single_candidate_all_tiers(
            rc=rc,
            repaired_txt=txt,
            orig_w=len(txt.split()),
            ledger_facts=facts,
            gate_info={"passed": True},
            comp=comp,
            p_map=p_map,
            aligner=aligner,
            vocab=None,
        )
        self.assertIsNotNone(res)
        tier_rank, new_rc, meta = res
        self.assertEqual(tier_rank, 1)
        self.assertEqual(meta["paras"], 2)
        self.assertGreaterEqual(meta["p2_novel"], 1)
        self.assertEqual(meta["n_1sup"], 0)
        self.assertIn("\n\n", new_rc["text"])

    def test_prefix_rescue_drops_bad_last_sentence_with_positive_tail_attribution(self):
        facts = {
            "f1": _make_fact("object", "positive", "dog", "a brown dog", ("supported", "supported"), 0.92),
            "f2": _make_fact("object", "positive", "frisbee", "a red frisbee", ("supported", "supported"), 0.90),
            "f3": _make_fact("object", "positive", "bench", "a wooden bench", ("supported", "supported"), 0.89),
            "f_bad": _make_fact("object", "positive", "umbrella", "a blue umbrella", ("unclear", "unclear"), 0.45),
        }
        comp = compile_strict_ledger(facts, {})
        p_map = build_p_map(facts, {})
        aligner = _StubAligner({
            "a brown dog": "f1",
            "a red frisbee": "f2",
            "a wooden bench": "f3",
            "a blue umbrella": "f_bad",
        })

        prefix_good = (
            "The image features a brown dog running energetically across an open grassy park while chasing a red frisbee on a sunny day. "
            "Bright afternoon sunlight illuminates the lush green lawn around the playful canine as it leaps forward across the field.\n\n"
            "In addition to the dog, a wooden bench sits quietly beside the paved walkway near the tall green shade trees."
        )
        full_with_bad_tail = prefix_good + " Further back in the distance, a blue umbrella is resting on the ground."
        rc = {
            "id": "img1:s1:0",
            "text": full_with_bad_tail,
            "facts": [
                {"quote": "a brown dog"},
                {"quote": "a red frisbee"},
                {"quote": "a wooden bench"},
                {"quote": "a blue umbrella"},
            ],
            "claim_ids": ["f1", "f2", "f3", "f_bad"],
            "fuzzy": {"u": 0.20},
        }
        res = inspect_single_candidate_all_tiers(
            rc=rc,
            repaired_txt=full_with_bad_tail,
            orig_w=len(full_with_bad_tail.split()),
            ledger_facts=facts,
            gate_info={"passed": True},
            comp=comp,
            p_map=p_map,
            aligner=aligner,
            vocab=None,
            min_words_prefix=60,
        )
        self.assertIsNotNone(res)
        tier_rank, new_rc, meta = res
        self.assertEqual(tier_rank, 2)
        self.assertEqual(new_rc["text"], prefix_good)
        self.assertEqual(new_rc["claim_ids"], ["f1", "f2", "f3"])
        self.assertEqual(meta["n_1sup"], 0)
        self.assertIn("\n\n", new_rc["text"])

        # 若坏命题位于前段而非被剔除句尾，正归属守卫必须拒绝！
        bad_in_front = (
            "The image features a brown dog and a blue umbrella on an open grassy park while chasing a red frisbee on a sunny day. "
            "Bright afternoon sunlight illuminates the lush green lawn around the playful canine as it leaps forward across the field.\n\n"
            "In addition to the dog, a wooden bench sits quietly beside the paved walkway near the tall green shade trees. "
            "The calm outdoor atmosphere makes the park look inviting."
        )
        rc_bad_front = dict(rc, text=bad_in_front)
        self.assertIsNone(
            inspect_single_candidate_all_tiers(
                rc=rc_bad_front,
                repaired_txt=bad_in_front,
                orig_w=len(bad_in_front.split()),
                ledger_facts=facts,
                gate_info={"passed": True},
                comp=comp,
                p_map=p_map,
                aligner=aligner,
                vocab=None,
                min_words_prefix=60,
            )
        )


if __name__ == "__main__":
    unittest.main()
