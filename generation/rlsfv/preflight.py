"""Read-only model-family checks; never load weights or start a GPU task."""
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

def inspect_environment(cfg):
    family = cfg.get('backend', 'llava')
    qwen = family in {'qwen', 'qwen2.5', 'qwen25', 'qwen3', 'qwen3-vl'}
    if not qwen and family != 'llava':
        raise ValueError('Unknown backend: ' + str(family))
    packages, missing, errors = {}, [], []
    required = ['torch', 'transformers', 'spacy', 'Pillow', 'numpy', 'nltk']
    if not qwen:
        required += ['peft', 'sentencepiece']
    for name in required:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
    paths = [Path(cfg['vendor']) / 'AMBER/inference.py']
    model_class = None
    if qwen:
        if not cfg.get('qwen_model'):
            errors.append('qwen_model missing')
        root = Path(cfg.get('qwen_model') or '__missing_qwen_model__')
        paths += [root / 'config.json', root / 'tokenizer_config.json', root / 'preprocessor_config.json']
        index = root / 'model.safetensors.index.json'
        if index.exists():
            paths.append(index)
            try:
                paths += [root / f for f in set(json.loads(index.read_text())['weight_map'].values())]
            except (ValueError, KeyError, TypeError) as exc:
                errors.append('Invalid model weight index: ' + str(exc))
        else:
            paths.append(root / 'model.safetensors')
        model_class = ('Qwen3VLForConditionalGeneration' if family in {'qwen3', 'qwen3-vl'}
                       else 'Qwen2_5_VLForConditionalGeneration')
        try:
            module = importlib.import_module('transformers')
            getattr(module, model_class)
            transformer_path = module.__file__
        except Exception as exc:
            transformer_path = None
            errors.append('Cannot import ' + model_class + ': ' + repr(exc))
    else:
        transformer_path = None
        paths += [Path(cfg['vendor']) / 'RLSF-V/train/llava_series/llava/model/builder.py',
                  Path(cfg['models']) / 'verified_files.json']
    files = {str(p): p.is_file() for p in paths}
    return {'passed': not missing and not errors and all(files.values()), 'backend': family,
            'python': sys.executable, 'packages': packages, 'missing_packages': missing,
            'files': files, 'errors': errors, 'model_class': model_class,
            'transformers_module': transformer_path, 'gpu_not_tested': True}
