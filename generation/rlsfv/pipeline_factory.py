"""One model-routing contract for generation, baselines and evaluation."""

def build_backend(cfg, output, variant='base', trained_adapter=None):
    family = cfg.get('backend', 'llava')
    if family in {'qwen', 'qwen2.5', 'qwen25', 'qwen3', 'qwen3-vl'}:
        if variant != 'base' or trained_adapter:
            raise ValueError('10.1 Qwen supports base generation/selection/evaluation only; '
                             'Qwen DPO and adapter loading are not implemented')
        if not cfg.get('qwen_model'):
            raise ValueError('Qwen requires config.qwen_model')
        try:
            from .qwen_backend import QwenBackend, Qwen3Backend
        except ModuleNotFoundError as exc:
            if exc.name != __package__ + '.qwen_backend':
                raise
            raise ValueError('Use the 10.1 Qwen package for this backend') from exc
        cls = Qwen3Backend if family in {'qwen3', 'qwen3-vl'} else QwenBackend
        return cls(cfg['qwen_model'], output, variant=variant, context_len=cfg.get('context_len'))
    if family != 'llava':
        raise ValueError('Unknown backend: ' + str(family))
    from .backend import LlavaBackend
    return LlavaBackend(cfg['models'], output, cfg['vendor'], variant=variant,
                        trained_adapter=trained_adapter, context_len=cfg.get('context_len'))

def build(cfg, output):
    from .backend import PosParser
    from .pipeline import Pipeline
    if cfg.get('extraction', {}).get('mode', 'original') not in {'original', 'structured_sentences', 'hybrid'}:
        raise ValueError('Unsupported extraction mode')
    return Pipeline(cfg, build_backend(cfg, output), PosParser(cfg['pos_model']), output)
