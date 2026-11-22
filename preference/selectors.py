"""冻结 20261031 线性特征契约（legacy_coco_v1）并新增 E3-V3 互补双路选优与对称长度困难负例对齐。"""
import math
try:
    import numpy as np
except ImportError:  # pragma: no cover - 兼容无 numpy 的轻量级预检/管理容器
    np = None

SCHEMA = "legacy_coco_v1"
ANCHOR = (0.09331491826301164, 0.022080104386646527, -0.8846049773503418)


def linear(model, x):
    if np is not None:
        x = np.asarray(x, dtype=float)
        mu = np.asarray(model['mu'])
        sd = np.asarray(model['sd'])
        w = np.asarray(model['coef'])
        if x.shape != mu.shape or sd.shape != mu.shape or w.shape != (len(x) + 1,):
            raise ValueError('Formula feature shape mismatch')
        if not all(np.isfinite(a).all() for a in (x, mu, sd, w)) or (sd <= 0).any():
            raise ValueError('Invalid formula values')
        return float(w[0] + ((x - mu) / sd) @ w[1:])
    xs = [float(v) for v in x]
    mus = [float(v) for v in model['mu']]
    sds = [float(v) for v in model['sd']]
    ws = [float(v) for v in model['coef']]
    if len(xs) != len(mus) or len(sds) != len(mus) or len(ws) != len(xs) + 1:
        raise ValueError('Formula feature shape mismatch')
    if not all(math.isfinite(v) for seq in (xs, mus, sds, ws) for v in seq) or any(s <= 0 for s in sds):
        raise ValueError('Invalid formula values')
    return float(ws[0] + sum(((xi - mi) / si) * wi for xi, mi, si, wi in zip(xs, mus, sds, ws[1:])))


def mix(c):
    return (c['coverage']['f'] + c['coverage']['s']) / 2


def qv(c):
    return float(c['coverage'].get('qv', 0.0) or 0.0)


def anchor_score(c):
    return ANCHOR[0] * c['ranks']['U'] / 22 + ANCHOR[1] * c['ranks']['f'] / 22 + ANCHOR[2] * c['unsupported_primary_fraction']


def product_score(c):
    return mix(c) * (0.05 + 0.95 * (c['p'] or 0) * (1 - c['unsupported_primary_fraction'])) ** 2


def product_score_v3(c):
    """E3-V3 引导轮友好乘积路线：用去重实体+关系核查覆盖 qv 融合共识 mix，释放 s1/s2 高覆盖真细节。"""
    cov_vr = 0.75 * qv(c) + 0.25 * mix(c)
    prec_term = (0.05 + 0.95 * (c['p'] or 0.0) * (1.0 - c['unsupported_primary_fraction'])) ** 2
    cert_term = max(0.5, 1.0 - 0.15 * float(c.get('u') or 0.0))
    return cov_vr * prec_term * cert_term


def risk_features(c, cons, cons2):
    return [
        c['u'], c['unsupported_primary_fraction'], c['words'], c['coverage'].get('qv', 0), c['p'] or 0,
        c['ranks']['U'] / 22, c['coverage']['f'], c['coverage']['s'], cons.get('obj_n', 0),
        cons.get('cons_sup_frac', 0), cons.get('freq_mean', 0), cons2.get('ext_sup_cov_frac', 0), cons2.get('no_ext_frac', 0),
    ]


def fallback_counts(c):
    if 'g_double_errors' not in c:
        raise ValueError('Missing effective G counts: re-export features from saved raw records with fix2; do not reuse raw double_errors for G fallback')
    effective = set(c['g_double_errors'])
    errors = set(c['errors'])
    if not effective <= errors:
        raise ValueError('Effective double refutations must be a subset of G errors')
    return (len(effective), len(errors))


def choose(im, risk_model, cons, cons2):
    """保留 V2 原始选对接口以兼容既有契约回归测试。"""
    pool = [c for c in im['candidates'] if not c['errors']]
    if not pool:
        least = min(fallback_counts(c) for c in im['candidates'])
        fallback = [c for c in im['candidates'] if fallback_counts(c) == least]
        return {'v1.2': max(fallback, key=anchor_score), 'product': max(fallback, key=product_score),
                'reason': 'G_fallback', 'fallback': True, 'fallback_risk_counts': list(least)}
    base = max(pool, key=anchor_score)
    iid = im['image_id']

    def risk(c):
        key = (iid, c['id'])
        if key not in cons or key not in cons2:
            raise ValueError('Missing consensus for G-pass candidate')
        z = linear(risk_model, risk_features(c, cons[key], cons2[key]))
        return 1 / (1 + math.exp(-max(-30, min(30, z))))

    eligible = [c for c in pool if risk(c) <= risk(base) + 0.02 and mix(c) >= mix(base) + 0.05]
    a = max(eligible, key=lambda c: (round(mix(c), 12), -risk(c))) if eligible else base
    b = max(pool, key=product_score)
    return {'v1.2': a, 'product': b, 'anchor': base, 'reason': 'normal', 'fallback': False}


def choose_v3(im, risk_model, cons, cons2):
    """E3-V3 互补双路线选优（不改冻结 13 维风险模型权重与输入顺序）：
    - v1.2 路线：在低风险安全集（Top-70% 低风险或 base+0.03）内最大化 (0.55*mix + 0.45*qv)；
    - product 路线：在全体零幻觉安全池中最大化去重核查实体+关系覆盖乘积 product_score_v3。
    """
    pool = [c for c in im['candidates'] if not c['errors']]
    if not pool:
        raise ValueError('choose_v3 requires a non-empty zero-error pool (no fallback permitted)')
    iid = im['image_id']

    def risk(c):
        key = (iid, c['id'])
        if key not in cons or key not in cons2:
            raise ValueError('Missing consensus for G-pass candidate')
        z = linear(risk_model, risk_features(c, cons[key], cons2[key]))
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    risks = {c['id']: risk(c) for c in pool}
    base = max(pool, key=anchor_score)
    sorted_r = sorted(risks.values())
    r_cap70 = sorted_r[min(len(sorted_r) - 1, max(0, int(math.ceil(0.70 * len(sorted_r))) - 1))]
    r_thresh = max(risks[base['id']] + 0.03, r_cap70)
    safe_pool = [c for c in pool if risks[c['id']] <= r_thresh]
    a = max(safe_pool, key=lambda c: (round(0.55 * mix(c) + 0.45 * qv(c), 12), round(c['p'] or 0.0, 6), -risks[c['id']]))
    b = max(pool, key=lambda c: (round(product_score_v3(c), 12), round(qv(c), 12), -risks[c['id']]))
    return {'v1.2': a, 'product': b, 'anchor': base, 'reason': 'v3_complementary_dual_route', 'fallback': False}


def _concat_diff(fr, fc):
    if np is not None:
        return np.concatenate([fr, fr - fc])
    return list(fr) + [ri - ci for ri, ci in zip(fr, fc)]


def reject_features(c, f):
    vals = [
        c['u'], c['unsupported_primary_fraction'], c['p'] or 0, c['words'],
        c['coverage'].get('qv', 0), c['coverage'].get('qF', 0), c['coverage']['f'], c['coverage']['s'],
        c['ranks']['U'] / 22, c['ranks']['f'] / 22, len(c['errors']), len(c['double_errors']),
        0,  # 历史模型拟合时该维恒为 0，严格保持 ABI
        c['stage'], int(c['branch'] == 'fresh'), int(bool(c['errors'])),
        f['object_n'], f['object_sup'], f['object_unc'], f['object_uns'], f['Do'],
        f['relation_n'], f['relation_uns'], 0, 0,
    ]
    return np.array(vals, dtype=float) if np is not None else [float(v) for v in vals]


def rejected(im, c, other, model, allf):
    pool = [r for r in im['candidates'] if r['id'] not in {c['id'], other['id']} and len(r['errors']) >= 2]
    if not pool:
        pool = [r for r in im['candidates'] if r['id'] not in {c['id'], other['id']} and r['errors']]
    if not pool:
        return None
    fc = reject_features(c, allf[im['image_id'] + '|' + c['id']])

    def score(r):
        fr = reject_features(r, allf[im['image_id'] + '|' + r['id']])
        return linear(model, _concat_diff(fr, fc))

    return max(pool, key=score)


def rejected_v3(im, c, other, model, allf, old_diag=None):
    """E3-V3 对称长度与双维覆盖率对齐负例选择：
    1. 优先选择含真实视觉反驳（definite G 错误或正向具象物体被反驳）的原始候选；若全图无显式反驳则回退至含零支持猜测/未决 or 且 p 严格劣于 chosen 的原始候选；
    2. 联合约束 delta_mix >= -0.028 与 delta_qv >= -0.028，彻底根除覆盖率倒挂；
    3. 非对称长度优先窗口 [0.80*W, 1.08*W] 并施加超长惩罚，抵消 50 维线性权重对词数的正偏置，使正负例均长保持对称。
    """
    iid = im['image_id']
    fc = reject_features(c, allf[iid + '|' + c['id']])
    pc = float(c.get('p') or 0.0)
    mc = mix(c)
    qvc = qv(c)
    wc = max(1.0, float(c['words']))

    hard_pool = []
    soft_pool = []
    for r in im['candidates']:
        if r['id'] in {c['id'], other['id']}:
            continue
        pr = float(r.get('p') or 0.0)
        d_old = (old_diag or {}).get(r['id']) or {}
        has_hard_err = bool(r['errors']) or d_old.get('any_ref_obj', 0) > 0
        has_soft_err = d_old.get('zero_sup_concrete', 0) > 0 or d_old.get('has_or', False)
        if has_hard_err and (pc - pr) > 1e-12:
            hard_pool.append(r)
        elif has_soft_err and (pc - pr) > 0.015:
            soft_pool.append(r)

    def cov_aligned(r):
        return (mc - mix(r)) >= -0.028 and (qvc - qv(r)) >= -0.028

    def score_v3(r):
        fr = reject_features(r, allf[iid + '|' + r['id']])
        base_s = linear(model, _concat_diff(fr, fc))
        w_ratio = (float(r['words']) - wc) / wc
        len_pen = 2.0 * max(0.0, w_ratio) + 0.5 * abs(w_ratio)
        err_bonus = 0.30 if len(r['errors']) >= 2 else (0.15 if len(r['errors']) == 1 else 0.0)
        return base_s + err_bonus - len_pen

    for base_pool in (hard_pool, soft_pool):
        if not base_pool:
            continue
        aligned = [r for r in base_pool if cov_aligned(r)]
        if not aligned:
            continue
        t1 = [r for r in aligned if 0.80 * wc <= r['words'] <= 1.08 * wc]
        t2 = [r for r in aligned if 0.72 * wc <= r['words'] <= 1.18 * wc]
        t3 = [r for r in aligned if 0.62 * wc <= r['words'] <= 1.32 * wc]
        chosen_tier = t1 or t2 or t3 or aligned
        return max(chosen_tier, key=score_v3)
    return None


def normalized_weights(rows):
    from collections import Counter
    counts = Counter(r['image_id'] for r in rows)
    for r in rows:
        r['pair_weight'] = len(rows) / (len(counts) * counts[r['image_id']])
    return rows


def deduplicate(rows):
    out = {}
    for r in rows:
        key = (r['image_id'], r['prompt'], r['chosen'].strip(), r['rejected'].strip())
        if key in out:
            out[key]['roles'] = sorted(set(out[key]['roles'] + r['roles']))
        else:
            out[key] = dict(r, roles=list(r['roles']))
    return normalized_weights(list(out.values()))
