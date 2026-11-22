"""Native HF multimodal I/O; algorithmic fuzzy scoring matches the legacy backend."""

import math, time, random
from pathlib import Path
from rlsfv.io import fingerprint, sha256
from rlsfv.runtime import ContextOverflow
from rlsfv.scoring import aggregate_fuzzy
from multimodel.settings import normalize_caption, verified_inventory


class SerialHFVisionBackend:
    def __init__(
        self, binding, workdir, variant="base", trained_adapter=None, context_len=None
    ):
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
            Qwen3VLForConditionalGeneration,
            AutoModelForImageTextToText,
        )

        self.torch = torch
        self.binding = binding
        self.profile = binding["profile"]
        if variant not in ("base", "trained"):
            raise ValueError("HF backend supports base or trained adapter")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        inventory = verified_inventory(binding["model_lock"])
        path = binding["model_dir"]
        family = self.profile["family"]
        kw = (
            {"max_pixels": self.profile["max_pixels"]}
            if "max_pixels" in self.profile
            else {}
        )
        self.processor = AutoProcessor.from_pretrained(
            path, local_files_only=True, **kw
        )
        if family == "qwen3_vl":
            self.processor.image_processor.size = dict(
                self.processor.image_processor.size, longest_edge=self.profile["max_pixels"]
            )
        if family == "internvl":
            if not hasattr(self.processor.image_processor, "max_patches"):
                raise RuntimeError(
                    "Pinned InternVL HF processor must expose max_patches"
                )
            self.processor.image_processor.max_patches = self.profile["max_patches"]
            self.processor.image_processor.crop_to_patches = True
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        cls = {
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
            "qwen3_vl": Qwen3VLForConditionalGeneration,
            "internvl": AutoModelForImageTextToText,
        }[family]
        self.model = cls.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            attn_implementation="sdpa",
        )
        hashes = {}
        if variant == "trained":
            from peft import PeftModel

            if not trained_adapter:
                raise ValueError("trained_adapter required")
            hashes = {
                p.name: sha256(p)
                for p in Path(trained_adapter).glob("*")
                if p.is_file()
            }
            self.model = PeftModel.from_pretrained(
                self.model, trained_adapter
            ).merge_and_unload()
        self.model.eval().requires_grad_(False)
        self.context_len = int(context_len or self.profile["context_len"])
        cfg = self.model.config.to_dict()
        capacity = cfg.get("text_config", cfg).get("max_position_embeddings")
        if capacity and self.context_len > capacity:
            raise ValueError("Context budget exceeds model capacity")
        self.meta = dict(
            variant=variant,
            adapter_sha256=hashes,
            model_inventory=inventory,
            tokenizer_hash=fingerprint(self.tokenizer.get_vocab()),
            system_template=self.processor.chat_template,
            history_policy="fresh single-turn",
            context_limit=self.context_len,
            model_config_hash=fingerprint(cfg),
            image_preprocessor=self.processor.image_processor.to_dict(),
            profile=self.profile,
            batch_execution="batched within GPU; row seed=group seed+row index; four independent GPU shards",
            normalization="training sampled captions only; raw_text preserved",
        )

    def _inputs(self, image_path, prompt):
        from PIL import Image

        contents = []
        images = []
        if image_path:
            with Image.open(image_path) as im:
                images = [im.convert("RGB")]
            contents.append({"type": "image"})
        contents.append({"type": "text", "text": prompt})
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": contents}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        kwargs = dict(text=[rendered], return_tensors="pt")
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(self.model.device)
        # Keep integral image grids/masks integral; cast only floating vision tensors.
        for k, v in inputs.items():
            if self.torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=self.model.dtype)
        return inputs, rendered

    def _output_limit(self, inputs, settings):
        limit = settings["max_new_tokens"]
        if limit is not None:
            return int(limit)
        prompt_length = self._budget(inputs, 1)
        return self.context_len - prompt_length

    def _budget(self, inputs, n):
        length = int(inputs["input_ids"].shape[1])
        if length + n > self.context_len:
            raise ContextOverflow(
                "Input plus output exceeds configured context; no silent truncation"
            )
        return length

    def prompt_length(self, image_path, prompt):
        return self._inputs(image_path, prompt)[0]["input_ids"].shape[1]

    def generate(self, image_path, prompt, settings, seed, capture_scores=False):
        import numpy as np

        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs, rendered = self._inputs(image_path, prompt)
        limit = self._output_limit(inputs, settings)
        n = self._budget(inputs, limit)
        temp = float(settings.get("temperature", 0))
        kw = dict(
            do_sample=temp > 0,
            temperature=temp if temp > 0 else None,
            top_p=settings.get("top_p") if temp > 0 else None,
            top_k=settings.get("top_k", 50) if temp > 0 else 50,
            max_new_tokens=limit,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=capture_scores,
        )
        start = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kw)
        tids = out.sequences[0, n:].tolist()
        raw = self.tokenizer.decode(
            tids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        text = normalize_caption(raw) if image_path and temp > 0 else raw
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        result = dict(
            text=text,
            raw_text=raw,
            normalization="markdown_v1" if text != raw else None,
            token_ids=tids,
            tokens=len(tids),
            output_token_budget=limit,
            prompt_tokens=n,
            seconds=time.monotonic() - start,
            hit_limit=len(tids) >= limit and tids[-1] not in eos,
            rendered_prompt_hash=fingerprint(rendered),
            context_messages=1,
            history_messages=0,
        )
        if capture_scores:
            rows = [s[0].float().topk(min(50, s.shape[-1])) for s in out.scores]
            result.update(
                native_topk_values=[
                    [v if math.isfinite(v) else None for v in r.values.tolist()]
                    for r in rows
                ],
                native_topk_indices=[r.indices.tolist() for r in rows],
            )
        return result

    def generate_batch(self, image_path, prompts, settings, seed):
        # Serial requests bound peak memory; seeds are explicit and recorded, no cross-request KV.
        return [
            self.generate(image_path, p, settings, (seed + i) % (2**32))
            for i, p in enumerate(prompts)
        ]

    def _letter_variants(self, letters):
        variants = {}
        for letter in letters:
            forms = [letter, " " + letter]
            if len(letter) > 1:
                forms += [letter.lower(), " " + letter.lower()]
            ids = {
                x[0]
                for f in forms
                if len(x := self.tokenizer.encode(f, add_special_tokens=False)) == 1
                and x[0] != self.tokenizer.unk_token_id
            }
            if not ids:
                raise ValueError("No single-token option " + letter)
            if any(ids & set(v) for v in variants.values()):
                raise ValueError("Overlapping option token sets")
            variants[letter] = sorted(ids)
        return variants

    def choice(self, prompt, letters, image_path=None):
        torch = self.torch
        inputs, rendered = self._inputs(image_path, prompt)
        n = self._budget(inputs, 1)
        variants = self._letter_variants(letters)
        start = time.monotonic()
        with torch.inference_mode():
            logits = (
                self.model(**inputs, use_cache=False, return_dict=True)
                .logits[0, -1]
                .float()
            )
        probs = torch.softmax(logits, -1)
        raw = {k: float(probs[v].sum()) for k, v in variants.items()}
        mass = sum(raw.values())
        return dict(
            probs={k: v / mass if mass > 0 else 0 for k, v in raw.items()},
            raw_probs=raw,
            mass=mass,
            token_variants=variants,
            argmax_token=self.tokenizer.convert_ids_to_tokens(int(logits.argmax())),
            prompt_tokens=n,
            seconds=time.monotonic() - start,
            rendered_prompt_hash=fingerprint(rendered),
            history_messages=0,
            context_messages=1,
            decoding="single forward pass; letter probabilities",
        )

    def choice_batch(self, prompts, letters, image_path=None):
        return [self.choice(p, letters, image_path) for p in prompts]

    def _choice_text(self, prompt, letters):
        return self.choice(prompt, letters)

    def generate_structured(self, prompt, settings, seed, schema):
        "Text-only extraction; grammar controls formatting, never truth."
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data,
        )
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.utils import GenerationMixin
        from .structured import schema_progress, make_token_enforcer

        if float(settings.get("temperature", 0)) != 0:
            raise ValueError("Structured extraction must use greedy decoding")
        torch = self.torch
        inputs, rendered = self._inputs(None, prompt)
        ids = inputs["input_ids"]
        limit = int(settings["max_new_tokens"])
        if ids.shape[1] + limit > self.context_len:
            raise ContextOverflow(
                "Extraction input plus budget exceeds context; no silent truncation"
            )
        if not hasattr(self, "_structured_tokenizer_data"):
            self._structured_tokenizer_data = build_token_enforcer_tokenizer_data(
                self.tokenizer
            )
        enforcer = make_token_enforcer(self._structured_tokenizer_data, schema)

        def allowed(batch_id, sent):
            return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens

        prefix_length = ids.shape[1]
        tokenizer = self.tokenizer

        class CompleteRoot(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(
                    input_ids[0, prefix_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                progress = schema_progress(text.strip(), schema)
                return progress["complete"] or progress["repeated"]

        started = time.monotonic()
        with torch.inference_mode():
            sequence = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=50,
                num_beams=1,
                max_new_tokens=limit,
                use_cache=True,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                prefix_allowed_tokens_fn=allowed,
                stopping_criteria=StoppingCriteriaList([CompleteRoot()]),
            )
        torch.cuda.synchronize()
        tokens = sequence[0, prefix_length:].tolist()
        text = self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()
        progress = schema_progress(text, schema)
        finished = progress["complete"]
        return {
            "text": text,
            "token_ids": tokens,
            "tokens": len(tokens),
            "prompt_tokens": prefix_length,
            "seconds": time.monotonic() - started,
            "hit_limit": len(tokens) >= limit and not finished,
            "json_complete": finished,
            "history_messages": 0,
            "context_messages": 1,
            "stop_reason": (
                "repeated_fact"
                if progress["repeated"]
                else (
                    "complete_json"
                    if finished
                    else "token_budget" if len(tokens) >= limit else "eos_incomplete"
                )
            ),
            "rendered_prompt_hash": fingerprint(rendered),
            "schema_hash": fingerprint(schema),
            "decoding": "lm-format-enforcer-0.11.3; fresh parser; complete JSON stop",
            "generation_overrides": {
                "do_sample": False,
                "num_beams": 1,
                "repetition_penalty": 1.0,
                "no_repeat_ngram_size": 0,
                "max_new_tokens": limit,
            },
            "semantic_validated": False,
        }

    def score(self, image_path, target_prompt, text, parser, config, sampling):
        from transformers import TopKLogitsWarper, TopPLogitsWarper

        torch = self.torch
        inputs, rendered = self._inputs(image_path, target_prompt)
        enc = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        token_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not token_ids:
            return {"u": None, "reason": "empty_answer"}
        if len(offsets) != len(token_ids):
            raise ValueError("Offset/token count mismatch")
        if (
            self.tokenizer.decode(
                token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            != text
        ):
            raise ValueError(
                "Tokenizer does not round-trip the answer; cannot claim exact span scoring"
            )
        prefix = inputs["input_ids"]
        n = prefix.shape[1]
        answer = torch.tensor([token_ids], dtype=prefix.dtype, device=prefix.device)
        inputs["input_ids"] = torch.cat([prefix, answer], dim=1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(answer)], dim=1
        )
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = torch.cat(
                [inputs["token_type_ids"], torch.zeros_like(answer)], dim=1
            )
        expanded_len = self._budget(inputs, 0)
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model(**inputs, use_cache=False, return_dict=True)
            logits = output.logits[0, n - 1 : -1].float()
            if logits.shape[0] != len(token_ids):
                raise ValueError("Shifted answer logits mismatch")
            logits = logits / float(sampling["temperature"])
            dummy = torch.zeros((1, 1), dtype=torch.long, device=logits.device)
            logits = TopKLogitsWarper(int(sampling["top_k"]))(dummy, logits)
            logits = TopPLogitsWarper(float(sampling["top_p"]))(dummy, logits)
            values = logits.topk(
                min(int(config["saved_top_k"]), logits.shape[-1]), dim=-1
            ).values
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
            term = torch.where(
                interior, -(m * torch.log2(m) + (1 - m) * torch.log2(1 - m)), 0
            )
            entropies = (
                torch.where(
                    (count > 1) & (scale > 0),
                    term.sum(-1) / count.clamp_min(1),
                    torch.zeros_like(scale),
                )
                .cpu()
                .tolist()
            )
            valid_values = valid.cpu().tolist()
        entropies = [x if valid else None for x, valid in zip(entropies, valid_values)]
        spans = parser.pos_spans(text, config["pos_tags"])
        mask = [
            any(a < end and b > start for a, b in spans) if end > start else False
            for start, end in offsets
        ]
        u = aggregate_fuzzy(entropies, mask, config["top_r"])
        torch.cuda.synchronize()
        return {
            "u": u,
            "entropies": entropies,
            "content_mask": mask,
            "token_offsets": offsets,
            "scored_text": text,
            "tokenizer_hash": self.meta["tokenizer_hash"],
            "token_ids": token_ids,
            "seconds": time.monotonic() - started,
            "prompt_tokens": expanded_len,
            "tokens": len(token_ids),
            "common_prompt": target_prompt,
            "common_prompt_hash": fingerprint(rendered),
            "history_messages": 0,
            "score_implementation": "common-context shifted teacher forcing; temperature/top-k/top-p then saved top-k; single-candidate rows "
            + (
                "H=0 (paper/author)"
                if config.get("single_candidate_entropy", "invalid") == "zero"
                else "invalid"
            ),
        }



# Canonical serial HuggingFace backend (paper-faithful, single-sequence inference)
HFVisionBackend = SerialHFVisionBackend
