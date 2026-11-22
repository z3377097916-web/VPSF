from pathlib import Path
from .io import SCHEMA, Store, atomic_json, read_json, fingerprint, sha256, code_hash, lock
from .config import X0, STAGES
from .facts import make_aligner, parse_facts, union, verification, parse_verification, caption_valid, ensure_existence
from .prompts import extraction, verify_prompt, checklist, guided
from .scoring import score, guard_refuted, effective_statuses, verified
from .runtime import ContextOverflow

LOW_MASS_LIMIT = 0.05


class Pipeline:
    def __init__(self, cfg, backend, parser, output, nlp=None):
        self.cfg, self.backend, self.parser = cfg, backend, parser
        mode = cfg.get('extraction', {}).get('mode', 'original')
        if mode not in {'original', 'structured_sentences', 'hybrid'}:
            raise ValueError('This distribution supports LLaVA-only extraction')
        self.output = Path(output)
        self.identity = {"config": cfg, "model": backend.meta, "code": code_hash()}
        self.hybrid = None
        if mode == 'hybrid':

            if nlp is None:
                import spacy
                nlp = spacy.load(cfg['extraction']['parser_model'])
            meta = getattr(nlp, "meta", {}) or {}
            self.identity["parser"] = {"model": cfg['extraction']['parser_model'],
                                       "meta": {k: meta.get(k) for k in ("name", "version", "spacy_version")}}
        self.store = Store(output, self.identity)
        self.aligner = make_aligner(cfg)
        if mode == 'hybrid':
            from .hybrid_extractor import HybridExtractor
            self.hybrid = HybridExtractor(backend, self.store, nlp, dict(cfg['extraction'], seed=cfg['seed']))
        self.compact = cfg.get('checklist', {}).get('mode', 'original') == 'compact'

    def generate(self, image, prompt, kind, request_id, validate):
        settings = self.cfg["generation"] if kind == "caption" else self.cfg["analysis"]
        image_hash = sha256(image) if image else None
        for attempt in range(self.cfg["retries"]+1):
            seed = int(fingerprint([self.cfg["seed"], request_id, attempt])[:8], 16)
            req = {"kind": kind, "image_sha256": image_hash, "prompt": prompt, "settings": settings,
                   "seed": seed, "request_id": request_id, "attempt": attempt, "history": []}
            result = self.store.cached(req, lambda: self.backend.generate(image, prompt, settings, seed))
            try:
                if result.get("history_messages", 0) != 0:
                    raise ValueError("Backend returned conversation history")
                if result.get("hit_limit"):
                    raise ValueError("TRUNCATED_OUTPUT")
                parsed = validate(result["text"])
                return parsed, result
            except (ValueError, TypeError, KeyError) as exc:
                self.store.event({"event": "format_failure", "request_id": request_id, "attempt": attempt, "error": str(exc)})
                deterministic = float(settings.get("temperature", 0)) == 0
                if deterministic and attempt < self.cfg["retries"]:


                    self.store.event({"event": "deterministic_retry_skipped", "request_id": request_id,
                                      "reason": "identical prompt/settings under greedy decoding"})
                if deterministic or attempt == self.cfg["retries"]:
                    raise ValueError(f"{kind} failed validation: {exc}") from exc
        raise AssertionError("unreachable")


    def batch_size(self, what):
        b = self.cfg.get("batching", {})
        return int(b.get("max_batch", 1)) if b.get(what, False) else 1

    def run_batched(self, requests, what, call_batch, call_single):
        """Serve cached requests, compute only missing ones (in batches when enabled).
        Every request is an independent one-turn prompt; batching never shares context."""
        out = [self.store.peek(r) for r in requests]
        missing = [i for i, r in enumerate(out) if r is None]
        size = self.batch_size(what)
        for start in range(0, len(missing), size):
            chunk = missing[start:start + size]
            if size > 1 and len(chunk) > 1 and call_batch is not None:
                results = call_batch([requests[i] for i in chunk])
            else:
                results = [call_single(requests[i]) for i in chunk]
            for i, res in zip(chunk, results):
                if res.get("history_messages", 0):
                    raise ValueError("Backend returned conversation history")
                out[i] = self.store.put(requests[i], res)
        return out

    def prompt_tokens(self, image, prompt):
        if hasattr(self.backend, "prompt_length"):
            return self.backend.prompt_length(image, prompt)
        from .questions import estimate_tokens
        return estimate_tokens(prompt) + 620

    def guided_prompt(self, image, previous, compared, stage, folder):
        "Build conditional questions. fix9 never trims the question list to fit."
        from .questions import compact_checklist, guided_compact, verified_facts
        ccfg = self.cfg.get("checklist") or {}
        guidance_facts = previous
        if ccfg.get("facts", "all") == "verified":
            guidance_facts = verified_facts(
                previous,
                min_support=float(ccfg.get("min_support", 1.0)),
                min_singleton_p=float(ccfg.get("min_singleton_p", 0.0)),
            )
        include_pending = bool(ccfg.get("include_pending", True))
        style = ccfg.get("prompt_style", "compact")
        open_detail = bool(ccfg.get("open_detail", False))
        max_entities = ccfg.get("max_entities")
        max_details = ccfg.get("max_details")
        open_observation = ccfg.get("open_observation", "full")
        relation_module = bool(ccfg.get("relation_module", False))
        relation_priority = bool(ccfg.get("relation_priority", False))
        limit = getattr(self.backend, "context_len", 2048) - int(self.cfg["generation"]["max_new_tokens"])
        tried = []
        levels = (0,) if ccfg.get("overflow_policy") == "stop" else (0, 1, 2)
        for level in levels:
            questions = compact_checklist(guidance_facts, compared, degrade=level,
                                          include_pending=include_pending, open_detail=open_detail,
                                          max_entities=max_entities, max_details=max_details,
                                          open_observation=open_observation,
                                          relation_priority=relation_priority,
                                          max_detail_questions=ccfg.get("max_detail_questions"),
                                          clean_entities=bool(ccfg.get("clean_entities", False)),
                                          refutation_priority=bool(ccfg.get("refutation_priority", False)))
            prompt = guided_compact(questions, style=style, relation_module=relation_module)
            n = self.prompt_tokens(image, prompt)
            tried.append({"level": level, "prompt_tokens_with_image": n, "limit": limit})
            if n <= limit:
                break
        atomic_json(folder / f"questions-{stage}.meta.json", {
            "degrade_level": level, "attempts": tried, "fact_source": ccfg.get("facts", "all"),
            "source_fact_count": len(previous), "included_fact_count": len(guidance_facts),
            "prompt_refinement": next((q["prompt_refinement"] for q in questions if "prompt_refinement" in q), None),
            "refutation_priority": bool(ccfg.get("refutation_priority", False)),
            "rendered_fact_count": len({k for q in questions if q.get("render") in {"existence", "detail", "global"} for k in q["claim_ids"]}),
            "rendered_entity_count": sum(q.get("render") == "existence" for q in questions),
            "rendered_detail_line_count": sum(q.get("render") in {"detail", "global"} for q in questions),
            "rendered_detail_question_count": sum(q.get("detail_question_count", 0)
                                                  for q in questions if q.get("render") in {"detail", "global"}),
            "omitted_questions": [{"question_id": q["question_id"], "reason": q.get("omission_reason", q["render"]),
                                   "claim_ids": q["claim_ids"]} for q in questions if q.get("render", "").startswith("omitted")],
            "overflow_policy": ccfg.get("overflow_policy", "legacy_degrade"),
            "context_fits": n <= limit})
        if ccfg.get("overflow_policy") == "stop" and n > limit:
            atomic_json(folder / f"questions-{stage}.json", questions)
            atomic_json(folder / f"questions-{stage}.prompt.json", {"prompt": prompt, "history": [],
                        "status": "CONTEXT_OVERFLOW", "prompt_tokens_with_image": n, "limit": limit})
            raise ContextOverflow(f"Complete question list requires {n} input tokens; actual model budget is {limit}. "
                                  "Image and all questions retained; no truncation or automatic pruning.")
        return questions, prompt

    def captions(self, image, record, stage, prompts):
        """Generate all slots of one round. Identical prompts are sampled together in one
        batch (one recorded seed per batch); invalid rows are regenerated one by one."""
        settings = self.cfg["generation"]
        image_hash = sha256(image)
        groups = {}
        for index, prompt in enumerate(prompts):
            groups.setdefault(prompt, []).append(index)
        results = [None] * len(prompts)
        size = self.batch_size("captions")
        for prompt, slots in groups.items():
            for start in range(0, len(slots), size):
                chunk = slots[start:start + size]
                if size > 1 and len(chunk) > 1 and hasattr(self.backend, "generate_batch"):
                    rid = f"{record['id']}:s{stage}:batch{'-'.join(map(str, chunk))}"
                    seed = int(fingerprint([self.cfg["seed"], rid, 0])[:8], 16)
                    req = {"kind": "caption_batch", "image_sha256": image_hash, "prompt": prompt, "settings": settings,
                           "seed": seed, "request_id": rid, "rows": len(chunk), "history": []}
                    rows = self.store.cached(req, lambda: self.backend.generate_batch(image, [prompt] * len(chunk), settings, seed))
                    for slot, raw in zip(chunk, rows):
                        try:
                            if raw.get("history_messages", 0) or raw.get("hit_limit"):
                                raise ValueError("TRUNCATED_OR_HISTORY")
                            results[slot] = (caption_valid(raw["text"]), dict(raw, batch_seed=seed))
                        except ValueError as exc:
                            self.store.event({"event": "batch_row_regenerated", "request_id": rid, "slot": slot, "error": str(exc)})
                            results[slot] = self.generate(image, prompt, "caption", f"{record['id']}:s{stage}:c{slot}:single",
                                                          caption_valid)
                else:
                    for slot in chunk:
                        results[slot] = self.generate(image, prompt, "caption", f"{record['id']}:s{stage}:c{slot}", caption_valid)
        return results

    def verify_all(self, image, facts):
        vcfg = self.cfg.get("verification")
        if vcfg:
            return self.verify_fast(image, facts, vcfg)
        for key, fact in sorted(facts.items()):
            checks = []
            for template in (0, 1):
                parsed, raw = self.generate(image, verify_prompt(fact, template), "verification",
                    f"verify-{key}-{template}", lambda text, t=template: verification(text, t))
                checks.append(dict(parsed, template=template, artifact_type="verification"))
            fact["checks"] = checks

    def verify_fast(self, image, facts, vcfg):
        "Two independent template checks per fact (+ optional soft read), batched."
        from .verification import (prompt_a, parse_a, in_scope, protocol_prompt, soft_values, template_b_settings,
                                   yesno_prompt, yesno_check, YESNO_WORDS)
        image_hash = sha256(image)
        todo = []
        for key, f in sorted(facts.items()):
            if in_scope(f, vcfg.get("scope", "all")):
                todo.append((key, f))
            else:
                f["checks"] = []
                f["verification_scope"] = "not_verified: hedged claim, unused by any score"
        cap = vcfg.get("template_b_max_new_tokens")
        settings = template_b_settings(self.cfg["analysis"], vcfg)

        def generated(template):
            reqs = [{"kind": "verification_" + ("A" if template == 0 else "B"), "image_sha256": image_hash,
                     "prompt": protocol_prompt(f, template), "settings": settings, "seed": 0, "history": []}
                    for _, f in todo]
            res = self.run_batched(reqs, "verification",
                lambda rs: self.backend.generate_batch(image, [r["prompt"] for r in rs], settings, 0),
                lambda r: self.backend.generate(image, r["prompt"], settings, 0))
            out = []
            for (key, f), r, q in zip(todo, res, reqs):
                if r.get("hit_limit") and cap is not None:
                    wide = dict(settings, max_new_tokens=max(256, int(cap) * 2))
                    r = self.store.cached(dict(q, settings=wide), lambda q=q, wide=wide: self.backend.generate(image, q["prompt"], wide, 0))
                    self.store.event({"event": "verification_extended_budget", "fact": key, "template": template})
                if r.get("hit_limit"):
                    raise ValueError("verification failed validation: TRUNCATED_OUTPUT")
                try:
                    verdict, normalized = parse_verification(r["text"], template)
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"verification failed validation: {exc}") from exc
                if normalized:
                    self.store.event({"event": "verification_format_normalized", "fact": key, "template": template})
                out.append(dict(verdict, template=template, method="generate", artifact_type="verification",
                                format_normalized=normalized))
            return out

        def soft(fs):
            built = [prompt_a(f) for f in fs]
            reqs = [{"kind": "verification_A_choice", "image_sha256": image_hash, "prompt": p,
                     "letters": sorted(m), "history": []} for p, m in built]
            res = self.run_batched(reqs, "verification",
                lambda rs: self.backend.choice_batch([r["prompt"] for r in rs], rs[0]["letters"], image),
                lambda r: self.backend.choice(r["prompt"], r["letters"], image))
            return [(r, m) for r, (_, m) in zip(res, built)]

        def yesno(template):

            reqs = [{"kind": "verification_%s_yesno" % ("A" if template == 0 else "B"), "image_sha256": image_hash,
                     "prompt": yesno_prompt(f, template), "letters": YESNO_WORDS, "history": []} for _, f in todo]
            res = self.run_batched(reqs, "verification",
                lambda rs: self.backend.choice_batch([r["prompt"] for r in rs], YESNO_WORDS, image),
                lambda r: self.backend.choice(r["prompt"], YESNO_WORDS, image))
            band, min_mass = vcfg.get("yesno_band", [0.35, 0.65]), vcfg.get("yesno_min_mass", 0.3)
            return [dict(yesno_check(r, f, band, min_mass), template=template, method="yesno")
                    for (_, f), r in zip(todo, res)]

        mode_a, mode_b = vcfg.get("template_a", "generate"), vcfg.get("template_b", "generate")
        if mode_a == "choice":
            checks_a = [dict(parse_a(r, m), template=0, method="choice", artifact_type="verification")
                        for r, m in soft([f for _, f in todo])]
        elif mode_a == "yesno":
            checks_a = yesno(0)
        else:
            checks_a = generated(0)
        checks_b = yesno(1) if mode_b == "yesno" else generated(1)

        if vcfg.get("soft_read", False) and mode_a != "yesno":
            reads = soft([f for _, f in todo])
        else:
            reads = [None] * len(todo)
        for (key, f), a, b, rd in zip(todo, checks_a, checks_b, reads):
            f["checks"] = [a, b]
            if rd is not None:
                f["soft_support"] = soft_values(*rd)
            elif vcfg.get("soft_read", False) and mode_a == "yesno":
                f["soft_support"] = {"supported": a["p_true"], "refuted": 1 - a["p_true"], "unclear": 0.0,
                                     "option_mass": a["option_mass"], "valid": not a["low_mass"],
                                     "source": "yesno_template_A"}

        yes_checks = [c for _, f in todo for c in f["checks"] if c.get("method") == "yesno"]
        if yes_checks:
            low = sum(c["low_mass"] for c in yes_checks)
            self.store.event({"event": "yesno_mass", "checks": len(yes_checks), "low_mass": low})
            if low / len(yes_checks) > LOW_MASS_LIMIT and not getattr(self, 'defer_subset_mass_review', False):
                raise ValueError("verification failed validation: LOW_MASS_REVIEW (%d of %d yes/no reads below "
                                 "yesno_min_mass)" % (low, len(yes_checks)))

    def foil_calibration(self, image, candidates, facts):
        """False-support rate of the soft read on claims that are almost surely false."""
        vcfg = self.cfg.get("verification") or {}
        n = int(vcfg.get("foils", 0))
        if not n or not vcfg.get("soft_read", False):
            return None
        from .verification import foil_facts, prompt_a, soft_values, yesno_prompt, yesno_check, YESNO_WORDS
        foils = foil_facts(candidates, facts, n_objects=n, n_colors=max(0, n - 1), seed=self.cfg["seed"])
        if not foils:
            return {"foils": [], "false_support_rate": None}
        def _rates(rows):
            by = {}
            for r in rows:
                if r['soft'].get('valid', True):
                    by.setdefault(r.get("foil"), []).append(r["soft"]["supported"])
            return ({k: sum(v) / len(v) for k, v in sorted(by.items())},
                    {k: len(v) for k, v in sorted(by.items())})
        image_hash = sha256(image)
        if vcfg.get("template_a", "generate") == "yesno":

            reqs = [{"kind": "verification_A_yesno", "image_sha256": image_hash, "prompt": yesno_prompt(f, 0),
                     "letters": YESNO_WORDS, "history": []} for f in foils]
            res = self.run_batched(reqs, "verification",
                lambda rs: self.backend.choice_batch([r["prompt"] for r in rs], YESNO_WORDS, image),
                lambda r: self.backend.choice(r["prompt"], YESNO_WORDS, image))
            band, min_mass = vcfg.get("yesno_band", [0.35, 0.65]), vcfg.get("yesno_min_mass", 0.3)
            rows = []
            for f, r in zip(foils, res):
                c = yesno_check(r, f, band, min_mass)
                rows.append(dict(f, soft={"supported": c["p_true"], "refuted": 1 - c["p_true"], "unclear": 0.0,
                                          "option_mass": c["option_mass"], "source": "yesno_template_A",
                                          "valid": not c['low_mass'], "low_mass": c['low_mass']}, raw_read=r))
            valid = [r for r in rows if r['soft']['valid']]
            rate = sum(r["soft"]["supported"] for r in valid) / len(valid) if valid else None
            by_type, counts = _rates(rows)
            return {"foils": rows, "false_support_rate": rate, "method": "yesno_template_A",
                    "false_support_rate_by_type": by_type, "foil_counts": counts,
                    "valid_count": len(valid), "invalid_count": len(rows) - len(valid),
                    "status": "available" if valid else "unavailable_all_reads_invalid",
                    "fallback": None if valid else "uncalibrated_support",
                    "note": "Foils are almost surely false (objects named by no caption, colour swapped to a colour no caption mentions); the rate is an upper-bound estimate of false support."}
        built = [prompt_a(f) for f in foils]
        reqs = [{"kind": "verification_A_choice", "image_sha256": image_hash, "prompt": p, "letters": sorted(m),
                 "history": []} for p, m in built]
        res = self.run_batched(reqs, "verification",
            lambda rs: self.backend.choice_batch([r["prompt"] for r in rs], rs[0]["letters"], image),
            lambda r: self.backend.choice(r["prompt"], r["letters"], image))
        rows = [dict(f, soft=soft_values(r, m)) for f, r, (_, m) in zip(foils, res, built)]
        rate = sum(r["soft"]["supported"] for r in rows) / len(rows)
        by_type, counts = _rates(rows)
        return {"foils": rows, "false_support_rate": rate,
                "false_support_rate_by_type": by_type, "foil_counts": counts,
                "note": "Foils are almost surely false (objects named by no caption, colour swapped to a colour no caption mentions); the rate is an upper-bound estimate of false support."}

    def image(self, record):
        image = str(Path(record["image"]).resolve())
        folder = self.output / "images" / fingerprint(record["id"])[:20]
        folder.mkdir(parents=True, exist_ok=True)
        input_identity = {"record": record, "image_sha256": sha256(image), "run": fingerprint(self.identity)}
        manifest_path = folder / "input.json"
        if manifest_path.exists() and read_json(manifest_path) != input_identity:
            raise ValueError("Existing image run has different inputs; choose new output directory")
        atomic_json(manifest_path, input_identity)
        done = folder / "result.json"
        if done.exists():
            return read_json(done)
        candidates, previous, snapshots = [], {}, []
        def state(name, **extra):
            atomic_json(folder / "status.json", {"status": name, "schema_version": SCHEMA, **extra})
        state("READY")
        try:
            for stage, count in enumerate(STAGES):
                state("GUIDING", stage=stage)
                compared = candidates
                if stage and self.cfg["prompt_policy"] == "current":
                    compared = [c for c in candidates if c["stage"] == stage-1]
                guided_text = None
                if stage and self.compact and self.cfg["prompt_policy"] != "generic":
                    questions, guided_text = self.guided_prompt(image, previous, compared, stage, folder)
                elif self.compact:
                    from .questions import compact_checklist
                    questions = compact_checklist(previous, compared) if stage else []
                else:
                    questions = checklist(previous, compared) if stage else []
                atomic_json(folder / f"questions-{stage}.json", questions)
                slots = []
                for index in range(count):
                    fresh = stage == 0 or (index == count-1 and self.cfg["prompt_policy"] != "all_guided")
                    if fresh:
                        prompt = X0
                    elif guided_text is not None:
                        prompt = guided_text
                    else:
                        prompt = guided(questions, self.cfg["prompt_policy"] == "generic")
                    slots.append((fresh, prompt))
                state("GENERATING", stage=stage)
                generated = self.captions(image, record, stage, [p for _, p in slots])
                for index, ((fresh, prompt), (text, raw)) in enumerate(zip(slots, generated)):
                    cid = f"{record['id']}:s{stage}:c{index}"
                    state("EXTRACTING", stage=stage, slot=index)
                    extra = {}
                    if self.hybrid is not None:
                        result = self.hybrid.extract(text)
                        facts, parsed_raw = result["facts"], None
                        extra = {"pending_queries": result["pending_queries"],
                                 "extraction": {k: result[k] for k in ("audit", "pending_facts", "rejected_facts", "uncovered", "entities")}}
                    elif self.cfg.get("extraction", {}).get("mode") == "structured_sentences":
                        from .structured import extract_caption
                        settings = dict(self.cfg["analysis"], max_new_tokens=self.cfg["extraction"]["max_new_tokens"])
                        facts, parsed_raw = extract_caption(self.backend, self.store, text, settings, self.cfg["seed"])
                    else:
                        facts, parsed_raw = self.generate(None, extraction(text), "claim_extraction", "extract-"+fingerprint(text),
                                                         lambda t: parse_facts(t, text))
                    c = {"id": cid, "stage": stage, "branch": "fresh" if fresh else "guided", "text": text,
                         "artifact_type": "caption", "prompt": prompt, "generation": raw, "facts": facts, **extra}
                    from .statement_guard import apply_candidate as guard_statements
                    c = guard_statements(c)

                    c['facts'] = ensure_existence(c['facts'])
                    if self.cfg.get("extraction", {}).get("source_scope_guard", False):
                        from .source_scope_guard import apply_candidate
                        c = apply_candidate(c)
                    candidates.append(c)
                    atomic_json(folder / f"candidate-{stage}-{index}.json", c)

                state("ALIGNING", stage=stage)
                cumulative = union(candidates, self.aligner)
                state("VERIFYING", stage=stage)
                self.verify_all(image, cumulative)
                atomic_json(folder / f"facts-{stage}.json", cumulative)
                snapshots.append({k: {"counts": f["counts"][:], "checks": f["checks"],
                                      "fresh_counts": f.get("fresh_counts"), "guided_counts": f.get("guided_counts"),
                                      "fresh_totals": f.get("fresh_totals"), "guided_totals": f.get("guided_totals")}
                                  for k, f in cumulative.items()})
                previous = cumulative
                if self.cfg["prompt_policy"] == "current":
                    current = union([c for c in candidates if c["stage"] == stage], self.aligner)
                    previous = {k: cumulative[k] for k in current}
            state("SCORING")
            for c in candidates:
                request = {"kind": "common_context_fuzzy", "image_sha256": input_identity["image_sha256"],
                           "prompt": X0, "caption": c["text"], "fuzzy": self.cfg["fuzzy"],
                           "sampling": self.cfg["generation"], "history": []}
                c["fuzzy"] = self.store.cached(request, lambda c=c: self.backend.score(
                    image, X0, c["text"], self.parser, self.cfg["fuzzy"], self.cfg["generation"]))
            dependencies=[]
            for key,f in cumulative.items():
                if f["category"] == "object": continue
                for obj_key,obj in cumulative.items():
                    if obj["category"] != "object" or obj["polarity"] != "positive": continue
                    if not any(e["anchor"] == oe["anchor"] and e["qualifiers"] == oe["qualifiers"] for e in f["entities"] for oe in obj["entities"]): continue
                    if guard_refuted(obj, self.cfg["scoring"]) and verified(f) and "supported" in effective_statuses(f):
                        dependencies.append({"claim_id":key,"existence_id":obj_key,"type":"dependency_conflict"})
            state("SELECTING")
            calibration = self.foil_calibration(image, candidates, cumulative)
            ranked = score(candidates, cumulative, self.cfg["scoring"], self.cfg["seed"], calibration)
            result = {"schema_version": SCHEMA, "status": "complete", "record": record,
                      "generation_identity": self.identity,
                      "image_sha256": input_identity["image_sha256"], "candidates": candidates,
                      "facts": cumulative, "snapshots": snapshots, "ranking": ranked, "dependency_conflicts": dependencies,
                      "foil_calibration": calibration}
            atomic_json(done, result)
            atomic_json(folder / "status.json", {"status": "complete"})
            return result
        except Exception as exc:
            error_state = "CONTEXT_OVERFLOW" if isinstance(exc, ContextOverflow) else "FORMAT_REVIEW" if "FORMAT_REVIEW" in str(exc) else "FAILED_REVIEW_REQUIRED"
            atomic_json(folder / "status.json", {"status": error_state, "error": repr(exc), "retained": True})
            return {"status": error_state, "record": record, "error": repr(exc)}

    def run(self, records):
        with lock(self.output):
            manifest = self.output / "run.json"
            identity = dict(self.identity, records=records)
            if manifest.exists() and read_json(manifest) != identity:
                raise ValueError("Config/code/input changed. Use a new run directory; do not mix runs")
            atomic_json(manifest, identity)
            results = []
            for i, record in enumerate(records):
                print(f"[{i+1}/{len(records)}] image {record['id']}", flush=True)
                r = self.image(record)
                results.append({"id": record["id"], "status": r["status"], "error": r.get("error")})
                atomic_json(self.output / "status.json", {"complete": all(x["status"] == "complete" for x in results)
                    and len(results) == len(records), "total": len(records), "images": results})
            return results
