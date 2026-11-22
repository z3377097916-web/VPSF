"""One caption -> one fresh request -> source checks -> review record.

Experimental component only. Does not alter the formal pipeline or AMBER scoring.
"""
import time
from . import anchored
from .io import fingerprint
from .syntax_audit import audit as syntax_audit


class SimpleFactExtractor:
    def __init__(self, backend, store, nlp=None, max_new_tokens=2048):
        self.backend = backend
        self.store = store
        self.nlp = nlp
        self.settings = {'temperature': 0, 'max_new_tokens': max_new_tokens}

    def extract(self, caption):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError('A nonempty original caption is required')
        started = time.monotonic()
        prompt = anchored.prompt(caption, caption)
        schema = anchored.schema(caption, caption)
        request = {
            'kind': 'simple_source_copy_extraction',
            'prompt': prompt,


            'schema_hash': fingerprint(schema),
            'settings': self.settings,
            'seed': 42,
            'history': [],
        }
        raw = self.store.cached(request, lambda: self.backend.generate_structured(
            prompt, self.settings, 42, schema))
        record = {
            'caption': caption,
            'caption_hash': fingerprint(caption),
            'raw': raw,
            'facts': [],
            'source_positions': [],
            'warnings': [],
            'eligible_for_formal_scoring': False,
            'semantic_acceptance': 'NOT_YET_VALIDATED',
            'status': 'REVIEW_REQUIRED',
            'coverage_audit': 'numeric_and_pos' if self.nlp else 'numeric_only',
        }
        try:
            if raw.get('history_messages', 0):
                raise ValueError('HISTORY_REVIEW: request must be independent')
            if raw.get('hit_limit') or raw.get('stop_reason') == 'repeated_fact':
                raise ValueError('INCOMPLETE_OR_REPEATED_OUTPUT_REVIEW')
            facts, positions = anchored.decode(raw['text'], caption, caption)
            record['facts'] = facts
            record['source_positions'] = positions
            record['warnings'] = anchored.audit(caption, caption, facts, self.nlp)
            if self.nlp is not None:
                record['syntax_warnings'] = syntax_audit(caption, facts, self.nlp)
                record['warnings'].extend(record['syntax_warnings'])
            if not record['warnings']:
                record['status'] = 'LOCAL_CHECKS_PASSED_SEMANTIC_REVIEW_PENDING'
        except (ValueError, TypeError, KeyError) as exc:
            record['error'] = str(exc)
        record['wall_seconds_including_cache_and_checks'] = time.monotonic() - started


        self.store.event({'event': 'simple_extraction_review',
                          'caption_hash': record['caption_hash'],
                          'status': record['status'],
                          'warnings': record['warnings'],
                          'error': record.get('error')})
        return record
