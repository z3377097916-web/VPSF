"""E3-V3 纯净度与双维覆盖率单调守卫（无目标测试集标注参与，保持对仅含 p 字段二次校验的向后兼容）。"""
import math

RULE = 'strict_p_joint_coverage_v3'
TOL = 1e-12
MIX_COV_TOL = -0.085  # 拦截严重共识覆盖率倒挂（delta_mix < -0.085），同时允许 75w 自然 s0 正例对比 106w 多错引导负例
QV_COV_TOL = -0.145   # 拦截极端核查实体覆盖倒挂，避免因负例盲目枚举多物体而误护多错负例


def valid(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1


def coverage(c):
    v = c.get('coverage') or {}
    f = v.get('f')
    s = v.get('s')
    qv = v.get('qv')
    fc = f if valid(f) else None
    sc = s if valid(s) else None
    mc = (f + s) / 2 if valid(f) and valid(s) else None
    qvc = qv if valid(qv) else None
    return fc, sc, mc, qvc


def evaluate(chosen, rejected):
    pc = chosen.get('p')
    pr = rejected.get('p')
    fc, sc, mc, qvc = coverage(chosen)
    fr, sr, mr, qvr = coverage(rejected)
    dm = (mc - mr) if (mc is not None and mr is not None) else None
    dqv = (qvc - qvr) if (qvc is not None and qvr is not None) else None
    has_cov_inputs = (dm is not None) or (dqv is not None)
    audit = dict(
        chosen_p=pc if valid(pc) else None,
        rejected_p=pr if valid(pr) else None,
        chosen_f=fc,
        chosen_s=sc,
        rejected_f=fr,
        rejected_s=sr,
        chosen_mix=mc,
        rejected_mix=mr,
        delta_mix=dm,
        chosen_qv=qvc,
        rejected_qv=qvr,
        delta_qv=dqv,
        chosen_cve=qvc,
        rejected_cve=qvr,
        delta_cve=dqv,
        coverage_direction_is_filter=has_cov_inputs,
    )
    if not valid(pc) or not valid(pr):
        return dict(passed=False, reason='missing_or_invalid_p', **audit)
    dp = pc - pr
    if dp <= TOL:
        reason = 'reject_hallucination_proxy_better' if dp < -TOL else 'no_strict_hallucination_proxy_advantage'
        return dict(passed=False, reason=reason, delta_p=dp,
                    chosen_hallucination_proxy=1 - pc, rejected_hallucination_proxy=1 - pr, **audit)
    if dm is not None and dm < MIX_COV_TOL:
        return dict(passed=False, reason='severe_mix_coverage_inversion', delta_p=dp,
                    chosen_hallucination_proxy=1 - pc, rejected_hallucination_proxy=1 - pr, **audit)
    if dqv is not None and dqv < QV_COV_TOL:
        return dict(passed=False, reason='severe_qv_coverage_inversion', delta_p=dp,
                    chosen_hallucination_proxy=1 - pc, rejected_hallucination_proxy=1 - pr, **audit)
    return dict(passed=True, reason='passed', delta_p=dp,
                chosen_hallucination_proxy=1 - pc, rejected_hallucination_proxy=1 - pr, **audit)
