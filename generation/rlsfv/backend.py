"""Reuse the pinned RLSF-V LLaVA loader; each call builds a fresh conversation."""
from __future__ import annotations

import importlib.metadata
import math
import json
from pathlib import Path
import sys
import time

from .scoring import aggregate_fuzzy
from .io import fingerprint, sha256, stable_model_config_hash
from .runtime import ContextOverflow, context_limit


def letter_variants(tokenizer, letters):
    """每个选项的 SentencePiece 写法：'▁A' 与 'A'。选项是多字母的词（Yes / No）时再加小写形式
    '▁yes'、'yes'（核查方式修订说明 3.1）；单个字母不加小写，因为 '▁a' 是冠词。
    分词表里不存在的写法跳过；一种写法都没有就报错。"""
    variants = {}
    for letter in letters:
        pieces = ["▁" + letter, letter]
        if len(letter) > 1 and letter.lower() != letter:
            pieces += ["▁" + letter.lower(), letter.lower()]
        cands = set()
        for piece in pieces:
            tid = tokenizer.convert_tokens_to_ids(piece)
            if tid is not None and tid != tokenizer.unk_token_id:
                cands.add(int(tid))
        if not cands:
            raise ValueError("Option letter has no single token: " + letter)
        variants[letter] = sorted(cands)
    return variants


def decoded_token_mask(tokenizer, token_ids, spans):
    """Map POS character spans to slow SentencePiece tokens by decoded prefixes."""
    ends = [0]
    for i in range(1, len(token_ids) + 1):
        ends.append(len(tokenizer.decode(token_ids[:i], skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)))
    if any(a > b for a, b in zip(ends, ends[1:])):
        raise ValueError("Nonmonotonic decoded token alignment; cannot silently score")
    return [any(start < ends[i + 1] and end > ends[i] for start, end in spans)
            for i in range(len(token_ids))]


class PosParser:
    def __init__(self, model):
        import spacy
        self.nlp = spacy.load(model, disable=["ner", "parser", "lemmatizer"])
    def pos_spans(self, text, tags):
        return [(t.idx, t.idx+len(t.text)) for t in self.nlp(text) if t.pos_ in tags]

class LlavaBackend:
    def __init__(self, models, workdir, vendor, variant="base", trained_adapter=None, context_len=None):
        import torch
        self.torch = torch
        models, workdir, vendor = Path(models).resolve(), Path(workdir).resolve(), Path(vendor).resolve()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; there is no automatic CPU or quantized fallback")
        if variant not in {"base", "rlsf", "trained"}:
            raise ValueError("Unknown model variant")
        if variant == "trained" and not trained_adapter:
            raise ValueError("--trained-adapter is required for a trained checkpoint")

        previous = vendor.parent
        manifest = json.loads((previous / "model_manifest.json").read_text())
        verified = json.loads((models / "verified_files.json").read_text())
        groups = {"base", "vision"} | ({"adapter"} if variant == "rlsf" else set())
        checked = set()
        for record in verified:
            if record["group"] not in groups:
                continue
            spec = manifest[record["group"]]
            entry = next(x for x in spec["files"] if x["name"] == record["name"])
            expected = (entry.get("lfs") or {}).get("sha256") or entry.get("official_sha256")
            path = models / record["group"] / record["name"]
            if not expected or record["revision"] != spec["sha"] or sha256(path) != expected:
                raise ValueError(f"Model SHA256/provenance verification failed: {path}")
            checked.add((record["group"], record["name"]))
        weight_index = json.loads((models / "base/pytorch_model.bin.index.json").read_text())
        required = {("base", n) for n in set(weight_index["weight_map"].values())} | {("vision", "pytorch_model.bin")}
        if variant == "rlsf":
            required |= {("adapter", "adapter_model.safetensors"), ("adapter", "non_lora_trainables.bin"), ("adapter", "config.json"), ("adapter", "adapter_config.json")}
        required |= {("base", f) for f in ("config.json", "generation_config.json", "pytorch_model.bin.index.json", "special_tokens_map.json", "tokenizer.model", "tokenizer_config.json")}
        required |= {("vision", "config.json"), ("vision", "preprocessor_config.json")}
        if not required <= checked:
            raise ValueError("Model verification inventory is incomplete")

        def alias(source, name):
            dest = workdir / "model_alias" / name
            dest.mkdir(parents=True, exist_ok=True)
            config = json.loads((source / "config.json").read_text())
            config["mm_vision_tower"] = str(models / "vision")
            for path in source.iterdir():
                if path.name == "config.json" or path.name.startswith("."):
                    continue
                target = dest / path.name
                if not target.exists():
                    target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
                elif target.resolve() != path.resolve():
                    raise ValueError("Existing model alias points to a different checkpoint")
            (dest / "config.json").write_text(json.dumps(config, indent=2))
            return dest

        base = alias(models / "base", "base")
        selected = base
        base_argument = None
        name = "llava-v1.5-13b" if json.loads((base / "config.json").read_text()).get("hidden_size") == 5120 else "llava-v1.5-7b"
        extra_hashes = {}
        if variant != "base":
            source = models / "adapter" if variant == "rlsf" else Path(trained_adapter).resolve()
            for filename in ("adapter_model.safetensors", "non_lora_trainables.bin", "adapter_config.json", "config.json"):
                if not (source / filename).is_file():
                    raise ValueError(f"Incomplete adapter: missing {source / filename}")
                extra_hashes[filename] = sha256(source / filename)
            selected = alias(source, "adapter")
            base_argument, name = str(base), "llava-v1.5-7b-lora"
        sys.path.insert(0, str(vendor / "RLSF-V/train/llava_series"))
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            str(selected), base_argument, name, device_map="auto", device="cuda",
            load_8bit=False, load_4bit=False, use_flash_attn=False)
        if any(str(p.device) == "cpu" for p in self.model.parameters()):
            raise RuntimeError("CPU offload detected; stop rather than changing the reference precision")
        if self.model.get_vision_tower().dtype != torch.float32:
            raise RuntimeError("Expected the reference float32 vision tower")
        loader_context_len = int(self.context_len)
        model_context_len = int(self.model.config.max_position_embeddings)
        self.context_len = context_limit(context_len, model_context_len, loader_context_len)
        self.model.config.tokenizer_model_max_length = None
        self.model.eval()
        self.model.requires_grad_(False)
        self.conv_templates, self.process_images = conv_templates, process_images
        self.tokenizer_image_token = tokenizer_image_token
        self.image_index = IMAGE_TOKEN_INDEX
        self.image_token = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            self.image_token = DEFAULT_IM_START_TOKEN + self.image_token + DEFAULT_IM_END_TOKEN
        self._image_path = None
        self._pixels = None
        self.meta = {"variant": variant, "adapter_sha256": extra_hashes, "model_inventory": verified,
                     "source_hashes": {str(p.relative_to(vendor)): sha256(p) for p in sorted((vendor / "RLSF-V/train/llava_series").rglob("*.py"))},
                     "tokenizer_hash": fingerprint(self.tokenizer.get_vocab()), "system_template": self.conv_templates["vicuna_v1"].get_prompt(),
                     "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda, "context_limit": self.context_len,
                     "context_requested": context_len, "context_model_capacity": model_context_len,
                     "context_loader_default": loader_context_len,
                     "vision_dtype": str(self.model.get_vision_tower().dtype),
                     "model_config_hash": stable_model_config_hash(self.model.config.to_dict()),
                     "image_preprocessor": self.image_processor.to_dict(),
                     "image_aspect_ratio": getattr(self.model.config,"image_aspect_ratio",None),
                     "attention": getattr(self.model.config, "_attn_implementation", None),
                     "packages": {x: importlib.metadata.version(x) for x in
                                  ("torch", "transformers", "spacy", "peft")},
                     "history_policy": "fresh system/user prompt per request; no prior assistant messages or external KV reuse"}

    def _inputs(self, image_path, prompt):
        from PIL import Image
        torch = self.torch
        if image_path is None:
            conv = self.conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            rendered = conv.get_prompt()
            ids = self.tokenizer(rendered, return_tensors="pt").input_ids.cuda()
            return ids, None, rendered
        path = str(Path(image_path).resolve())

        if path != self._image_path:
            with Image.open(path) as raw:
                image = raw.convert("RGB")
                self._size = image.size
                pixels = self.process_images([image], self.image_processor, self.model.config)[0]
            self._pixels = pixels.unsqueeze(0).to(device="cuda", dtype=torch.float16)
            self._image_path = path
        conv = self.conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], self.image_token + "\n" + prompt)
        conv.append_message(conv.roles[1], None)
        rendered = conv.get_prompt()
        ids = self.tokenizer_image_token(rendered, self.tokenizer, self.image_index,
                                         return_tensors="pt").unsqueeze(0).cuda()
        return ids, self._pixels, rendered

    def generate(self, image_path, prompt, settings, seed, capture_scores=False):
        import random
        import numpy as np
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ids, pixels, rendered = self._inputs(image_path, prompt)

        projected_length = ids.shape[1] + (self.model.get_vision_tower().num_patches - 1 if image_path else 0)
        limit = int(settings["max_new_tokens"])
        if projected_length + limit > self.context_len:
            raise ContextOverflow("Prompt plus requested output exceeds context window; no silent truncation")
        temperature = float(settings.get("temperature", 0))
        kwargs = {"do_sample": temperature > 0, "temperature": temperature if temperature > 0 else None,
                  "top_p": settings.get("top_p") if temperature > 0 else None,
                  "top_k": settings.get("top_k", 50) if temperature > 0 else 50,
                  "num_beams": 1, "max_new_tokens": limit, "use_cache": True}
        if capture_scores:
            kwargs.update(return_dict_in_generate=True, output_scores=True)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(ids, **({"images": pixels, "image_sizes": [self._size]} if image_path else {}), **kwargs)
        torch.cuda.synchronize()
        native = {}
        if capture_scores:
            rows = [step[0].float().topk(min(50, step.shape[-1])) for step in output.scores]
            native = {"native_topk_values": [[v if math.isfinite(v) else None for v in row.values.cpu().tolist()] for row in rows],
                      "native_topk_indices": [row.indices.cpu().tolist() for row in rows]}
            token_ids = output.sequences[0].tolist()
        else:
            token_ids = output[0].tolist()
        if len(token_ids) > limit:
            raise ValueError("Unexpected prompt prefix in generated output")
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
        eos = self.tokenizer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        return {**native, "text": text, "tokens": len(token_ids), "token_ids": token_ids,
                "prompt_tokens": projected_length, "seconds": time.monotonic() - started,
                "hit_limit": len(token_ids) >= limit and token_ids[-1] not in eos_ids,
                "rendered_prompt_hash": fingerprint(rendered), "context_messages": 1,
                "history_messages": 0}

    def generate_structured(self, prompt, settings, seed, schema):
        "Text-only extraction; grammar controls formatting, never truth."
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data)
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.utils import GenerationMixin
        from .structured import schema_progress, make_token_enforcer
        if float(settings.get("temperature", 0)) != 0:
            raise ValueError("Structured extraction must use greedy decoding")
        torch = self.torch
        ids, _, rendered = self._inputs(None, prompt)
        limit = int(settings["max_new_tokens"])
        if ids.shape[1] + limit > self.context_len:
            raise ContextOverflow("Extraction input plus budget exceeds context; no silent truncation")
        if not hasattr(self, "_structured_tokenizer_data"):
            self._structured_tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        enforcer = make_token_enforcer(self._structured_tokenizer_data, schema)
        def allowed(batch_id, sent):
            return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
        prefix_length = ids.shape[1]
        tokenizer = self.tokenizer
        class CompleteRoot(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0, prefix_length:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
                progress = schema_progress(text.strip(), schema)
                return progress["complete"] or progress["repeated"]
        started = time.monotonic()
        with torch.inference_mode():
            sequence = GenerationMixin.generate(self.model, input_ids=ids,
                attention_mask=torch.ones_like(ids), do_sample=False, temperature=None,
                top_p=None, top_k=50, num_beams=1, max_new_tokens=limit, use_cache=True,
                repetition_penalty=1.0, no_repeat_ngram_size=0,
                prefix_allowed_tokens_fn=allowed,
                stopping_criteria=StoppingCriteriaList([CompleteRoot()]))
        torch.cuda.synchronize()
        tokens = sequence[0, prefix_length:].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
        progress = schema_progress(text, schema)
        finished = progress["complete"]
        return {"text": text, "token_ids": tokens, "tokens": len(tokens),
                "prompt_tokens": prefix_length, "seconds": time.monotonic()-started,
                "hit_limit": len(tokens) >= limit and not finished,
                "json_complete": finished, "history_messages": 0, "context_messages": 1,
                "stop_reason": "repeated_fact" if progress["repeated"] else "complete_json" if finished else "token_budget" if len(tokens) >= limit else "eos_incomplete",
                "rendered_prompt_hash": fingerprint(rendered), "schema_hash": fingerprint(schema),
                "decoding": "lm-format-enforcer-0.11.3; fresh parser; complete JSON stop",
                "generation_overrides": {"do_sample": False, "num_beams": 1,
                    "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
                    "max_new_tokens": limit},
                "semantic_validated": False}

    def prompt_length(self, image_path, prompt):
        """Exact input length (image placeholder expanded) for budget checks before generation."""
        ids, _, _ = self._inputs(image_path, prompt)
        return int(ids.shape[1]) + (self.model.get_vision_tower().num_patches - 1 if image_path else 0)


    def _left_pad(self, rows):
        """Left-pad several fresh single-turn prompts (no shared context between rows)."""
        torch = self.torch
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id
        lengths = [r.shape[1] for r in rows]
        width = max(lengths)
        ids = torch.full((len(rows), width), pad, dtype=rows[0].dtype, device=rows[0].device)
        mask = torch.zeros((len(rows), width), dtype=torch.long, device=rows[0].device)
        for i, r in enumerate(rows):
            ids[i, width - r.shape[1]:] = r[0]
            mask[i, width - r.shape[1]:] = 1
        return ids, mask, lengths

    def _images(self, pixels, n):
        return {"images": pixels.expand(n, -1, -1, -1).contiguous(), "image_sizes": [self._size] * n}

    def generate_batch(self, image_path, prompts, settings, seed):
        "Independent requests decoded together. Each row is its own fresh one-turn"
        import random
        import numpy as np
        torch = self.torch
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        built = [self._inputs(image_path, p) for p in prompts]
        ids, mask, lengths = self._left_pad([b[0] for b in built])
        pixels = built[0][1]
        limit = int(settings["max_new_tokens"])
        extra = (self.model.get_vision_tower().num_patches - 1) if image_path else 0
        if ids.shape[1] + extra + limit > self.context_len:
            raise ContextOverflow("Batched prompt plus requested output exceeds context; no silent truncation")
        temperature = float(settings.get("temperature", 0))
        kwargs = {"do_sample": temperature > 0, "temperature": temperature if temperature > 0 else None,
                  "top_p": settings.get("top_p") if temperature > 0 else None,
                  "top_k": settings.get("top_k", 50) if temperature > 0 else 50,
                  "num_beams": 1, "max_new_tokens": limit, "use_cache": True,
                  "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id}
        previous = getattr(self.model.config, "tokenizer_padding_side", "right")
        self.model.config.tokenizer_padding_side = "left"
        started = time.monotonic()
        try:
            with torch.inference_mode():
                out = self.model.generate(ids, attention_mask=mask,
                                          **(self._images(pixels, len(prompts)) if image_path else {}), **kwargs)
            torch.cuda.synchronize()
        finally:
            self.model.config.tokenizer_padding_side = previous
        eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        seconds = time.monotonic() - started
        results = []
        for i in range(len(prompts)):
            row = out[i].tolist()
            cut = next((j + 1 for j, t in enumerate(row) if t in eos_ids), None)
            tokens = row[:cut] if cut else row
            text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
            results.append({"text": text, "tokens": len(tokens), "token_ids": tokens,
                            "prompt_tokens": lengths[i] + extra, "seconds": seconds / len(prompts),
                            "batch_seconds": seconds, "batch_size": len(prompts), "batch_row": i,
                            "hit_limit": cut is None and len(tokens) >= limit,
                            "history_messages": 0, "context_messages": 1,
                            "rendered_prompt_hash": fingerprint(built[i][2])})
        return results

    def choice_batch(self, prompts, letters, image_path=None):
        """Option-letter probabilities for several independent prompts in one forward pass."""
        torch = self.torch
        built = [self._inputs(image_path, p) for p in prompts]
        ids, mask, lengths = self._left_pad([b[0] for b in built])
        extra = (self.model.get_vision_tower().num_patches - 1) if image_path else 0
        if ids.shape[1] + extra + 1 > self.context_len:
            raise ContextOverflow("Choice prompt exceeds context; no silent truncation")
        variants = self._letter_variants(letters)
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        previous = getattr(self.model.config, "tokenizer_padding_side", "right")
        self.model.config.tokenizer_padding_side = "left"
        started = time.monotonic()
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False,
                                    return_dict=True, **(self._images(built[0][1], len(prompts)) if image_path else {})
                                    ).logits[:, -1].float()
        finally:
            self.model.config.tokenizer_padding_side = previous
        probs_full = torch.softmax(logits, dim=-1)
        seconds = time.monotonic() - started
        out = []
        for i in range(len(prompts)):
            raw = {k: float(sum(probs_full[i, t] for t in v)) for k, v in variants.items()}
            mass = sum(raw.values())
            out.append({"probs": {k: (v / mass if mass > 0 else 0.0) for k, v in raw.items()}, "raw_probs": raw,
                        "mass": mass, "token_variants": variants, "prompt_tokens": lengths[i] + extra,
                        "seconds": seconds / len(prompts), "batch_size": len(prompts), "batch_row": i,
                        "rendered_prompt_hash": fingerprint(built[i][2]), "history_messages": 0,
                        "context_messages": 1, "decoding": "single forward pass; letter probabilities"})
        return out

    def _letter_variants(self, letters):
        return letter_variants(self.tokenizer, letters)

    def choice(self, prompt, letters, image_path=None):
        if image_path is not None:
            return self.choice_batch([prompt], letters, image_path)[0]
        return self._choice_text(prompt, letters)

    def _choice_text(self, prompt, letters):
        "Text-only multiple choice: one forward pass, probabilities of the option letters."
        torch = self.torch
        ids, _, rendered = self._inputs(None, prompt)
        if ids.shape[1] + 1 > self.context_len:
            raise ContextOverflow("Adjudication prompt exceeds context; no silent truncation")
        variants = self._letter_variants(letters)
        started = time.monotonic()
        with torch.inference_mode():
            logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                use_cache=False, return_dict=True).logits[0, -1].float()
        probs_full = torch.softmax(logits, dim=-1)
        raw = {k: float(sum(probs_full[t] for t in v)) for k, v in variants.items()}
        mass = sum(raw.values())
        probs = {k: (v / mass if mass > 0 else 0.0) for k, v in raw.items()}
        top = int(torch.argmax(logits))
        return {"probs": probs, "raw_probs": raw, "mass": mass, "token_variants": variants,
                "argmax_token": self.tokenizer.convert_ids_to_tokens(top),
                "prompt_tokens": int(ids.shape[1]), "seconds": time.monotonic() - started,
                "rendered_prompt_hash": fingerprint(rendered), "history_messages": 0,
                "context_messages": 1, "decoding": "single forward pass; letter probabilities"}

    def score(self, image_path, target_prompt, text, parser, config, sampling):
        from transformers import TopKLogitsWarper, TopPLogitsWarper
        torch = self.torch
        prefix, pixels, rendered = self._inputs(image_path, target_prompt)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return {"u": None, "reason": "empty_answer"}
        answer = torch.tensor([token_ids], dtype=prefix.dtype, device=prefix.device)
        full = torch.cat([prefix, answer], dim=1)
        labels = torch.full_like(full, -100)
        labels[:, prefix.shape[1]:] = answer
        expanded_len = full.shape[1] + self.model.get_vision_tower().num_patches - 1
        if expanded_len > self.context_len:
            raise ContextOverflow("Scoring sequence exceeds context; no silent truncation")
        started = time.monotonic()
        with torch.inference_mode():
            ids, positions, attention, past, embeds, expanded_labels = self.model.prepare_inputs_labels_for_multimodal(
                full, None, torch.ones_like(full, dtype=torch.bool), None, labels, pixels, [self._size])
            output = self.model(input_ids=ids, position_ids=positions, attention_mask=attention,
                                past_key_values=None, inputs_embeds=embeds, use_cache=False, return_dict=True)

            keep = expanded_labels[0, 1:] != -100
            logits = output.logits[0, :-1][keep].float()
            if logits.shape[0] != len(token_ids):
                raise ValueError("Image expansion / shifted token scoring alignment failed")
            logits = logits / float(sampling["temperature"])
            dummy = torch.zeros((1, 1), dtype=torch.long, device=logits.device)
            logits = TopKLogitsWarper(int(sampling["top_k"]))(dummy, logits)
            logits = TopPLogitsWarper(float(sampling["top_p"]))(dummy, logits)
            values = logits.topk(min(int(config["saved_top_k"]), logits.shape[-1]), dim=-1).values
            finite = torch.isfinite(values)
            count = finite.sum(-1)
            hi = values.masked_fill(~finite, float("-inf")).max(-1).values
            lo = values.masked_fill(~finite, float("inf")).min(-1).values
            scale = hi - lo
            if config.get("single_candidate_entropy", "invalid") == "zero":


                valid = (count >= 1) & ~torch.isnan(values).any(dim=-1)
            else:
                valid = (count > 1) & (scale > 0) & ~torch.isnan(values).any(dim=-1)
            mu = (values - lo[:, None]) / scale.clamp_min(1e-20)[:, None]
            interior = finite & (mu > 0) & (mu < 1)
            m = torch.where(interior, mu, 0.5)
            term = torch.where(interior, -(m * torch.log2(m) + (1 - m) * torch.log2(1 - m)), 0)
            entropies = torch.where((count > 1) & (scale > 0), term.sum(-1) / count.clamp_min(1),
                                    torch.zeros_like(scale)).cpu().tolist()
            valid_values = valid.cpu().tolist()
        entropies = [x if valid else None for x, valid in zip(entropies, valid_values)]
        pieces = self.tokenizer.convert_ids_to_tokens(token_ids)
        pieces = [t.replace("▁", " ").replace("<0x0A>", "\n") for t in pieces]
        pieces = ["" if t.startswith("<") and t.endswith(">") else t for t in pieces]
        spans = parser.pos_spans("".join(pieces), config["pos_tags"])
        mask, offset = [], 0
        for piece in pieces:
            end = offset + len(piece)
            mask.append(any(a < end and b > offset for a, b in spans) if end > offset else False)
            offset = end
        u = aggregate_fuzzy(entropies, mask, config["top_r"])
        torch.cuda.synchronize()
        return {"u": u, "entropies": entropies, "content_mask": mask,
                "token_ids": token_ids, "seconds": time.monotonic() - started,
                "prompt_tokens": expanded_len, "tokens": len(token_ids),
                "common_prompt": target_prompt, "common_prompt_hash": fingerprint(rendered), "history_messages": 0,
                "score_implementation": "common-context shifted teacher forcing; temperature/top-k/top-p then saved top-k; single-candidate rows " + ("H=0 (paper/author)" if config.get("single_candidate_entropy", "invalid") == "zero" else "invalid")}
