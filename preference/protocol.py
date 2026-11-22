"""运行时自反馈边界契约（开发集公式拟合独立披露，运行时零外部真值访问）。"""
from generation.rlsfv.task_contract import X0 as PROMPT

POLICY = 'e3_structural_repair_v3'
FEATURE_FILES = ('all_fact_features.json', 'consensus_features.json', 'consensus_features2.json')
GROUPS = ('E3_V3',)
FEEDBACK_BOUNDARY = {
    'formula_parameters_fitted_on_labeled_development_data': True,
    'parameters_frozen_during_pair_construction': True,
    'pair_construction_uses_external_annotations': False,
    'pair_construction_uses_external_feedback_model': False,
    'pair_construction_uses_human_feedback': False,
    'runtime_feedback_source': 'original_model_saved_candidates_and_self_verification',
    'coverage_is_model_proxy_not_ground_truth': True,
}
