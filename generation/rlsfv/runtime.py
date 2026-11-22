"""Backend-neutral runtime checks; no changes to the experiment's scoring rules."""


class ContextOverflow(RuntimeError):
    pass


def context_limit(requested, model_limit, legacy_limit=None):
    """Respect an explicit budget without inventing a model's context capacity."""
    if type(model_limit) is not int or model_limit < 1:
        raise ValueError('Missing/invalid model context capacity')
    if requested is not None:
        if type(requested) is not int or requested < 1 or requested > model_limit:
            raise ValueError(f'context_len must be an integer in [1, {model_limit}]')
        return requested
    return min(model_limit, legacy_limit) if legacy_limit else model_limit
