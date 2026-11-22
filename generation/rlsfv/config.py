from pathlib import Path
from .io import read_json
from .task_contract import X0

CATEGORIES = ("object", "attribute", "count", "spatial", "action", "relation",
              "text_ocr", "identity", "scene", "style", "other")
STAGES = (10, 7, 5)

def load(path):
    cfg = read_json(path)
    required = {"seed", "models", "vendor", "generation", "analysis", "fuzzy", "scoring",
                "synonyms", "retries", "prompt_policy", "pos_model"}
    if required - cfg.keys():
        raise ValueError(f"Missing config: {sorted(required-cfg.keys())}")
    for name in ("models", "vendor"):
        cfg[name] = str((Path(path).resolve().parent / cfg[name]).resolve())
    if cfg.get('qwen_model'):
        cfg['qwen_model'] = str((Path(path).resolve().parent / cfg['qwen_model']).resolve())
    if cfg.get('context_len') is not None and (type(cfg['context_len']) is not int or cfg['context_len'] < 1):
        raise ValueError('context_len must be a positive integer or null')
    if cfg["prompt_policy"] not in {"cumulative", "current", "all_guided", "generic"}:
        raise ValueError("Unknown prompt policy")
    if cfg["retries"] not in range(3):
        raise ValueError("Retries must be 0, 1 or 2")
    for section in ("generation", "analysis"):
        v = cfg[section]
        if set(v) != {"temperature", "top_p", "top_k", "max_new_tokens"}:
            raise ValueError(f"Explicit decoding parameters required: {section}")
        if v["temperature"] < 0 or not 0 < v["top_p"] <= 1 or v["top_k"] < 2 or v["max_new_tokens"] < 1:
            raise ValueError("Invalid decoding configuration")
    if cfg["generation"]["temperature"] <= 0:
        raise ValueError("Candidate sampling requires positive temperature")
    if cfg["analysis"]["temperature"] != 0:
        raise ValueError("Extraction and verification must be greedy")
    if "extraction" in cfg:
        e = cfg["extraction"]
        if e.get("mode") == "hybrid":
            expected = {"mode", "parser_model", "adjudicate", "reject_threshold", "min_option_mass", "calibrate", "claim_check"}
            optional = {"fidelity_audit", "pronoun_resolution", "direction_consistency", "source_scope_guard"}
            if not expected <= set(e) <= expected | optional:
                raise ValueError("Hybrid extraction needs explicit " + ", ".join(sorted(expected)))
            for key in ("fidelity_audit", "pronoun_resolution", "direction_consistency", "source_scope_guard"):
                if type(e.get(key, False)) is not bool:
                    raise ValueError("extraction.%s must be bool" % key)
            if e["adjudicate"] not in {"risky", "all", "none"} or not 0.5 <= e["reject_threshold"] <= 1 \
                    or not 0 <= e["min_option_mass"] < 1 or type(e["calibrate"]) is not bool \
                    or type(e["claim_check"]) is not bool:
                raise ValueError("Invalid hybrid extraction settings")
        else:
            expected = {"mode", "max_new_tokens"}
            if set(e) != expected or e["mode"] not in {"original", "structured_sentences"}:
                raise ValueError("This distribution supports LLaVA-only extraction")
            if type(e["max_new_tokens"]) is not int or e["max_new_tokens"] < 1:
                raise ValueError("Extraction token budget must be a positive integer")
    v = cfg.get("verification")
    if v is not None:
        cap = v.get("template_b_max_new_tokens")
        required = {"template_a", "template_b_max_new_tokens", "scope", "soft_read", "foils"}

        optional = {"template_b", "yesno_band", "yesno_min_mass"}
        if not required <= set(v) or not set(v) <= required | optional \
                or v["template_a"] not in {"generate", "choice", "yesno"} or v["scope"] not in {"scored", "all"} \
                or not (cap is None or (type(cap) is int and cap >= 16)) or type(v["soft_read"]) is not bool \
                or type(v["foils"]) is not int or v["foils"] < 0:
            raise ValueError("verification needs template_a generate|choice|yesno, template_b_max_new_tokens null|>=16, "
                             "scope scored|all, soft_read bool, foils int>=0 (optional: template_b, yesno_band, yesno_min_mass)")
        if v.get("template_b", "generate") not in {"generate", "yesno"}:
            raise ValueError("verification.template_b must be generate|yesno")
        band = v.get("yesno_band", [0.35, 0.65])
        if not (isinstance(band, list) and len(band) == 2 and all(type(x) in (int, float) for x in band)
                and 0 <= band[0] <= band[1] <= 1):
            raise ValueError("verification.yesno_band must be [lo, hi] with 0 <= lo <= hi <= 1")
        mm = v.get("yesno_min_mass", 0.3)
        if type(mm) not in (int, float) or not 0 <= mm < 1:
            raise ValueError("verification.yesno_min_mass must be in [0, 1)")
    b = cfg.get("batching")
    if b is not None and (set(b) != {"captions", "verification", "max_batch"} or type(b["max_batch"]) is not int
                          or b["max_batch"] < 1 or any(type(b[k]) is not bool for k in ("captions", "verification"))):
        raise ValueError("batching needs captions, verification (bool) and max_batch (int)")
    if cfg.get("alignment", {}).get("mode", "conservative") not in {"conservative", "lemma"}:
        raise ValueError("Unknown alignment mode")
    cl = cfg.get("checklist", {})
    if cl.get("mode", "original") not in {"original", "compact"}:
        raise ValueError("Unknown checklist mode")
    optional = {"facts", "min_support", "include_pending", "prompt_style",
                "open_detail", "max_entities", "max_details", "open_observation",
                "relation_module", "relation_priority", "overflow_policy", "max_detail_questions",
                "clean_entities", "refutation_priority"}
    if not set(cl) <= {"mode"} | optional:
        raise ValueError("checklist has unknown keys: %s" % sorted(set(cl) - {"mode"} - optional))
    if cl.get("facts", "all") not in {"all", "verified"}:
        raise ValueError("checklist.facts must be all|verified")
    ms = cl.get("min_support", 1.0)
    if type(ms) not in (int, float) or not 0 <= ms <= 1:
        raise ValueError("checklist.min_support must be in [0, 1]")
    if type(cl.get("include_pending", True)) is not bool:
        raise ValueError("checklist.include_pending must be bool")
    if cl.get("prompt_style", "compact") not in {"compact", "verified", "conditional"}:
        raise ValueError("checklist.prompt_style must be compact|verified|conditional")
    if cl.get("overflow_policy", "legacy_degrade") not in {"legacy_degrade", "stop"}:
        raise ValueError("checklist.overflow_policy must be stop|legacy_degrade")
    if type(cl.get("open_detail", False)) is not bool:
        raise ValueError("checklist.open_detail must be bool")
    for key in ("max_entities", "max_details", "max_detail_questions"):
        if key in cl and (type(cl[key]) is not int or cl[key] < 1):
            raise ValueError("checklist.%s must be a positive integer" % key)
    if cl.get("open_observation", "full") not in {"full", "short"}:
        raise ValueError("checklist.open_observation must be full|short")
    for key in ("relation_module", "relation_priority", "clean_entities", "refutation_priority"):
        if type(cl.get(key, False)) is not bool:
            raise ValueError("checklist.%s must be bool" % key)
    s = cfg["scoring"]
    base = {"alpha", "eta", "lambda", "rho", "pool", "strict_q", "class_balanced"}
    optional = {"exclude_refuted", "exclude_subjective", "rejected_rule", "soft_q", "foil_calibration",
                "coverage_mode", "frequency_source", "guard_rule", "bs_fallback_rule"}
    if not base <= set(s) <= base | optional:
        raise ValueError("All scoring parameters must be explicit")
    if s.get("coverage_mode", "legacy_categories") not in {"legacy_categories", "object_relation"}:
        raise ValueError("scoring.coverage_mode must be legacy_categories|object_relation")
    if s.get("frequency_source", "all") not in {"all", "fresh"}:
        raise ValueError("scoring.frequency_source must be all|fresh")
    if s.get("guard_rule", "any_refuted") not in {"any_refuted", "double_refuted"}:
        raise ValueError("scoring.guard_rule must be any_refuted|double_refuted")
    if s.get("bs_fallback_rule", "legacy_score") not in {"legacy_score", "min_double_then_any"}:
        raise ValueError("scoring.bs_fallback_rule must be legacy_score|min_double_then_any")
    if s.get("frequency_source") == "fresh" and cfg["prompt_policy"] == "all_guided":
        raise ValueError("Fresh-only statistics require a fresh caption in each round")
    if s.get("coverage_mode") == "object_relation" and not s.get("class_balanced"):
        raise ValueError("Object/relation coverage uses fixed equal group weights; class_balanced must be true")
    if any(type(s.get(k, False)) is not bool for k in ("exclude_refuted", "exclude_subjective", "soft_q", "foil_calibration")) \
            or s.get("rejected_rule", "lowest") not in {"lowest", "length_matched"}:
        raise ValueError("Invalid scoring switches")
    if s["alpha"] < 0 or not 0 <= s["eta"] <= 1 or s["lambda"] < 0 or not 0 < s["rho"] <= 1 or s["pool"] not in (22, 12, 10, 5):
        raise ValueError("Invalid scoring configuration")
    f = cfg["fuzzy"]
    if f.get("single_candidate_entropy", "invalid") not in {"invalid", "zero"}:
        raise ValueError("fuzzy.single_candidate_entropy must be invalid or zero")
    if f["top_r"] < 1 or f["saved_top_k"] < 2 or not f["pos_tags"]:
        raise ValueError("Invalid fuzzy configuration")
    return cfg
