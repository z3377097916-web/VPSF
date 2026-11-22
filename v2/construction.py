"""E3-V3 偏好对构造管线：复用独立 M1 证据账本，融合外科修复、具象边缘风险门控与对称长度双路选对。"""
from copy import deepcopy
from collections import defaultdict
from rlsfv.facts import union, ensure_existence, caption_valid
from rlsfv.scoring import score, any_refuted
from rlsfv.statement_guard import apply_candidate
from rlsfv.verification import yesno_prompt
from rlsfv.config import X0
from rlsfv.io import fingerprint
from scripts.extract_features import compact_record, extra, status
from preference.selectors import choose_v3, rejected_v3
from preference.purity import evaluate, RULE
from preference.protocol import PROMPT
from .repair import repair, POLICY
from .ledger_guard import candidate_v3_diagnostics, is_concrete_entity
from rlsfv.scoring import effective_statuses


def _has_contaminated_fact(cand_facts: dict) -> bool:
    """检测修复后事实集中是否仍残留被反驳命题或零支持具象物体（用于受保护跨度的闭环回退）。"""
    for f in cand_facts.values():
        ss = effective_statuses(f)
        if f.get('modality') == 'definite' and 'refuted' in ss:
            return True
        if f.get('polarity') == 'positive' and f.get('category') in {'object', 'identity'}:
            if 'refuted' in ss:
                return True
            conc, _ = is_concrete_entity(f)
            if conc and ss.count('supported') == 0:
                return True
    return False


def verify_reusing_exact(pipeline, image, facts, source_facts):
    """仅当视觉问题文本、极性、图像与基座模型完全一致时复用已有双模板核查结果。"""
    missing = {}
    index = {}
    for f in source_facts.values():
        checks = f.get('checks', [])
        if len(checks) == 2 and all(c.get('method') == 'yesno' for c in checks):
            key = (f['polarity'], yesno_prompt(f, 0), yesno_prompt(f, 1))
            index[key] = f
    reused = 0
    for k, f in facts.items():
        key = (f['polarity'], yesno_prompt(f, 0), yesno_prompt(f, 1))
        old = index.get(key)
        if old is None:
            missing[k] = f
        else:
            f['checks'] = deepcopy(old['checks'])
            if 'soft_support' in old:
                f['soft_support'] = deepcopy(old['soft_support'])
            f['verification_reuse'] = 'identical_original_model_image_questions'
            reused += 1
    if missing:
        pipeline.verify_all(image, missing)
    return dict(reused=reused, new=len(missing))


def extract_repaired(pipeline, original, text, audit, image, v2_cand_by_id=None):
    if text == original['text']:
        return deepcopy(original)
    # 若与经过同图像同源哈希校验的 V2 修复文本逐字一致，直接安全复用其事实抽取与公共前缀模糊熵
    if v2_cand_by_id is not None:
        v2_c = v2_cand_by_id.get(original['id'])
        if v2_c is not None and v2_c.get('text') == text and v2_c.get('fuzzy', {}).get('common_prompt') == X0:
            reused_c = deepcopy(v2_c)
            reused_c['repair'] = dict(
                policy=POLICY, original_text=original['text'], audit=audit,
                original_generation_record_is_not_repaired_logits=True,
                reused_identical_v2_extraction=True,
            )
            return reused_c
    result = pipeline.hybrid.extract(caption_valid(text))
    c = {k: deepcopy(v) for k, v in original.items()
         if k not in {'facts', 'claim_ids', 'fuzzy', 'pending_queries', 'extraction', 'statement_guard'}}
    c.update(
        text=text, facts=result['facts'], pending_queries=result['pending_queries'],
        extraction={k: result[k] for k in ('audit', 'pending_facts', 'rejected_facts', 'uncovered', 'entities')},
        repair=dict(policy=POLICY, original_text=original['text'], audit=audit,
                    original_generation_record_is_not_repaired_logits=True),
    )
    c = apply_candidate(c)
    c['facts'] = ensure_existence(c['facts'])
    if pipeline.cfg.get('extraction', {}).get('source_scope_guard', False):
        from rlsfv.source_scope_guard import apply_candidate as source_guard
        c = source_guard(c)
    req = dict(kind='v2_repaired_common_context_fuzzy', image_sha256=pipeline.image_hash,
               prompt=X0, caption=text, fuzzy=pipeline.cfg['fuzzy'], sampling=pipeline.cfg['generation'], history=[])
    c['fuzzy'] = pipeline.store.cached(req, lambda: pipeline.backend.score(
        image, X0, text, pipeline.parser, pipeline.cfg['fuzzy'], pipeline.cfg['generation']))
    return c


def original_consensus(new_result, source_result, nlp=None):
    """保持冻结 13 维风险模型外部一致性特征语义不变。"""
    def objects(c, facts):
        out = {}
        for k in c['claim_ids']:
            f = facts[k]
            if f['modality'] != 'definite' or f['polarity'] != 'positive' or f['category'] != 'object':
                continue
            name = ' '.join(str(f.get('subject') or '').lower().split())
            st = status(f)
            if name and (name not in out or out[name] != 'supported' and st == 'supported'):
                out[name] = st
        return out

    eligible = {r['id'] for r in source_result['ranking']['rows'] if r['eligible']}
    originals = {c['id']: objects(c, source_result['facts']) for c in source_result['candidates'] if c['id'] in eligible}
    holders = defaultdict(dict)
    for cid, obj in originals.items():
        for name, st in obj.items():
            holders[name][cid] = st
    cons, cons2 = {}, {}
    iid = str(new_result['record']['id'])
    for c in new_result['candidates']:
        obj = objects(c, new_result['facts'])
        den = max(1, len(obj))
        multi = len(originals) > 1
        both = sum(sum(v == 'supported' for v in holders[name].values()) >= 2 for name in obj) if multi else 0
        freq = sum(len(holders[name]) for name in obj) / den if multi else 0
        ext = no_ext = 0
        if multi:
            for name, st in obj.items():
                supported = any(cid != c['id'] and v == 'supported' for cid, v in holders[name].items())
                ext += int(st == 'supported' and supported)
                no_ext += int(not supported)
        cons[iid, c['id']] = dict(obj_n=len(obj), cons_sup_frac=both / den, freq_mean=freq)
        cons2[iid, c['id']] = dict(ext_sup_cov_frac=ext / den, no_ext_frac=no_ext / den)
    return cons, cons2


def naming_checks(pipeline, image, candidate, ledger, original_facts):
    """跨模态正向对象/身份反驳门控（含 pending 挂起命题双模板补核查）。"""
    accepted = {k: ledger[k] for k in candidate['claim_ids']
                if ledger[k]['polarity'] == 'positive' and ledger[k]['category'] in {'object', 'identity'}}
    pending = {}
    for f in candidate.get('extraction', {}).get('pending_facts', []) + candidate.get('statement_guard', {}).get('pending_facts', []):
        if f.get('polarity') == 'positive' and f.get('category') in {'object', 'identity'} and f.get('statement'):
            pending[fingerprint(f)] = deepcopy(f)
    if pending:
        verify_reusing_exact(pipeline, image, pending, {**original_facts, **ledger})
    errors = [
        dict(id=k, source=source, statement=f['statement'], modality=f['modality'], checks=f['checks'])
        for source, fs in [('accepted', accepted), ('pending', pending)]
        for k, f in fs.items() if any_refuted(f)
    ]
    return dict(passed=not errors, errors=errors, pending_checks=pending,
                rule='positive_object_or_identity_any_valid_refutation_regardless_of_modality')


def construct(source, entry, pipeline, nlp, risk, reject_models, v2_artifact=None):
    image = entry['image']
    old_candidates = deepcopy(source['candidates'])
    old_facts = deepcopy(source['facts'])
    if source['image_sha256'] != entry['image_sha256']:
        raise ValueError('Source candidates do not belong to this image')
    from preference.source_contract import raw_source
    source_contract, source_run_hash = raw_source(source)

    v2_cand_by_id = None
    v2_ledger_seed = deepcopy(old_facts)
    if v2_artifact is not None and v2_artifact.get('image_sha256') == entry['image_sha256'] and v2_artifact.get('source_sha256') == entry['source_sha256']:
        v2_cand_by_id = {c['id']: c for c in v2_artifact.get('candidates', [])}
        v2_ledger_seed = {**old_facts, **(v2_artifact.get('facts') or {})}
        for g in (v2_artifact.get('naming_gates') or {}).values():
            for pk, pf in (g.get('pending_checks') or {}).items():
                v2_ledger_seed.setdefault(pk, pf)

    old_im = compact_record(source)
    old_feat_by_id = {f['id']: f for f in old_im['candidates']}

    cs = []
    edit_count = 0
    for original in old_candidates:
        cid = original['id']
        old_f = old_feat_by_id[cid]
        # 阶段 A：若原始候选已 100% 通过金牌零幻觉/零 or/零否定/低边际风险准入（含 pending 命名门控），则无损保留原句
        pre_diag = candidate_v3_diagnostics(old_f, old_facts, None, max_r_edge=0.45)
        if pre_diag['strict_no_or_safe']:
            orig_gate = naming_checks(pipeline, image, original, old_facts, v2_ledger_seed)
            for pk, pf in (orig_gate.get('pending_checks') or {}).items():
                v2_ledger_seed.setdefault(pk, pf)
            if candidate_v3_diagnostics(old_f, old_facts, orig_gate, max_r_edge=0.45)['strict_no_or_safe']:
                cs.append(deepcopy(original))
                continue

        # 阶段 B：证据账本保护下的外科修复；仅当 guard 实际阻止了裁剪且 GPU 核查发现污染时闭环回退至严格裁剪
        text, audit = repair(nlp, original['text'], image, pipeline,
                             claim_ids=original.get('claim_ids'), source_facts=old_facts, use_guard=True)
        c = extract_repaired(pipeline, original, text, audit, image, v2_cand_by_id=v2_cand_by_id)
        if any(a.get('kind') == 'ledger_guard_protected_span' for a in audit):
            text_ug, audit_ug = repair(nlp, original['text'], image, pipeline,
                                       claim_ids=None, source_facts=None, use_guard=False)
            if text_ug != text:
                tmp_ledger = union([deepcopy(c)], pipeline.aligner)
                verify_reusing_exact(pipeline, image, tmp_ledger, v2_ledger_seed)
                for tk, tf in tmp_ledger.items():
                    v2_ledger_seed.setdefault(tk, tf)
                if _has_contaminated_fact(tmp_ledger):
                    audit_fb = list(audit_ug) + [dict(kind='closed_loop_contaminated_guard_fallback', guarded_text=text)]
                    text = text_ug
                    c = extract_repaired(pipeline, original, text_ug, audit_fb, image, v2_cand_by_id=v2_cand_by_id)

        edit_count += text != original['text']
        cs.append(c)

    new_facts = union(cs, pipeline.aligner)
    ledger = deepcopy(old_facts)
    for k, f in new_facts.items():
        if k not in ledger:
            ledger[k] = deepcopy(f)
            ledger[k].update(
                counts=[0, 0, 0], fresh_counts=[0, 0, 0], guided_counts=[0, 0, 0],
                fresh_totals=[10, 1, 1], guided_totals=[0, 6, 4], candidates=[],
                original_evidence_absent=True,
            )
    verification = verify_reusing_exact(pipeline, image, ledger, v2_ledger_seed)
    evidence = [dict(id=c['id'], branch=c['branch'], stage=c['stage'], claim_ids=c['claim_ids']) for c in old_candidates]
    ranked = score(cs, ledger, pipeline.cfg['scoring'], pipeline.cfg['seed'], source.get('foil_calibration'),
                   evidence_candidates=evidence)
    result = dict(status='complete', record=dict(source['record'], image=image), image_sha256=entry['image_sha256'],
                  candidates=cs, facts=ledger, evidence_candidates=evidence, ranking=ranked)
    new_im = compact_record(result)
    new_allf, _, _ = extra(result)
    old_allf, _, _ = extra(source)
    cons, cons2 = original_consensus(result, source)

    byid = {c['id']: c for c in cs}
    gates = {}
    for f in new_im['candidates']:
        if not f['errors']:
            gates[f['id']] = naming_checks(pipeline, image, byid[f['id']], ledger, v2_ledger_seed)

    review_facts = list(ledger.values()) + [f for g in gates.values() for f in g['pending_checks'].values()]
    visual_checks = [check for f in review_facts for check in f.get('checks', [])]
    low_mass = sum(bool(c.get('low_mass')) for c in visual_checks)
    if visual_checks and low_mass / len(visual_checks) > .05:
        raise ValueError('LOW_MASS_REVIEW over complete image ledger, including pending naming checks')

    # 计算修复池与原始池的 V3 具象零支持、否定缺失与边缘风险诊断
    v3_diag = {
        c['id']: candidate_v3_diagnostics(c, ledger, gates.get(c['id']), max_r_edge=0.45)
        for c in new_im['candidates']
    }
    old_diag = {
        c['id']: candidate_v3_diagnostics(c, ledger, None, max_r_edge=0.45)
        for c in old_im['candidates']
    }

    # 双层零幻觉无 or 准入：优先金牌零模糊池（strict_no_or_safe），全图金牌为空时启用银牌单模糊救援池（仍 100% 零反驳、零 or、零否定从句）
    strict_no_or = [c for c in new_im['candidates'] if v3_diag[c['id']]['strict_no_or_safe']]
    silver_no_or = [c for c in new_im['candidates'] if v3_diag[c['id']]['silver_no_or_safe']]
    allowed = strict_no_or if strict_no_or else silver_no_or
    pairs = []
    construction_audit = []
    if allowed:
        sel = choose_v3(dict(new_im, candidates=allowed), risk, cons, cons2)
        assert not sel['fallback']
        for role, other in [('v1.2', 'product'), ('product', 'v1.2')]:
            chosen = sel[role]
            mix_features = dict(old_allf)
            mix_features[str(entry['id']) + '|' + chosen['id']] = new_allf[str(entry['id']) + '|' + chosen['id']]
            rej = rejected_v3(old_im, chosen, sel[other], reject_models[role]['model'], mix_features, old_diag=old_diag)
            audit = dict(role=role, chosen_id=chosen['id'], rejected_id=None if rej is None else rej['id'],
                         chosen_v3_diag=v3_diag[chosen['id']])
            if rej is None:
                audit['reason'] = 'no_aligned_original_reject'
            else:
                purity = evaluate(chosen, rej)
                audit.update(purity=purity, reason=purity['reason'])
                if purity['passed'] and chosen['text'].strip() != rej['text'].strip():
                    pairs.append(dict(
                        image_id=str(entry['id']), img_path=image, image_sha256=entry['image_sha256'],
                        prompt=PROMPT, chosen=chosen['text'], rejected=rej['text'],
                        chosen_id=chosen['id'], rejected_id=rej['id'], roles=[role],
                        data_policy=POLICY, feature_schema='legacy_coco_v1_with_original_evidence',
                        g_fallback=False, chosen_G_errors=0, rejected_G_errors=len(rej['errors']),
                        naming_gate_passed=True, purity_rule=RULE, purity_checks=[purity],
                        source_sha256=entry['source_sha256'], source_contract=source_contract,
                        repaired=bool(byid[chosen['id']].get('repair')),
                        original_chosen_text=next(c['text'] for c in old_candidates if c['id'] == chosen['id']),
                    ))
            construction_audit.append(audit)

    result.update(
        policy=POLICY, source_contract=source_contract, source_run_hash=source_run_hash,
        source_file=entry['source'], source_sha256=entry['source_sha256'],
        verification=verification, naming_gates=gates, v3_diagnostics=v3_diag,
        edited_candidates=edit_count, eligible_candidates=len(allowed), pairs=pairs,
        construction_audit=construction_audit,
        skip_reason=None if pairs else 'no_safe_candidate' if not allowed else 'no_strict_pair',
        original_features=old_im, repaired_features=new_im,
        original_calibration=source.get('foil_calibration'),
        image_mass_review=dict(checks=len(visual_checks), low_mass=low_mass, threshold=.05),
    )
    return result
