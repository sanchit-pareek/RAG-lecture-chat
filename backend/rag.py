"""Shared implementation for API, CLI, indexing and evaluation."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(os.getenv('RAG_ROOT', str(Path(__file__).resolve().parents[1])))

def corpus():
    rows, lectures, digest = [], [], hashlib.sha256()
    files = sorted((ROOT / 'transcripts').glob('*.json'))
    if not files:
        raise ValueError('No transcripts found')
    seen = set()
    for path in files:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b'\0' + raw)
        doc = json.loads(raw)
        lecture = doc['lecture']
        if lecture in seen:
            raise ValueError(f'Duplicate lecture: {lecture}')
        seen.add(lecture)
        lectures.append(dict(lecture=lecture, duration=float(doc['duration']), bytes=len(raw)))
        for local_id, seg in enumerate(doc['segments']):
            start, end = float(seg['start']), float(seg['end'])
            text = seg['text'].strip()
            if not text or not np.isfinite([start, end]).all() or not 0 <= start <= end:
                raise ValueError(f'Invalid segment: {path.name}:{local_id}')
            rows.append(dict(lecture=lecture, local_segment_id=local_id, start=start, end=end, text=text))
    frame = pd.DataFrame(rows).sort_values(['lecture', 'start', 'local_segment_id']).reset_index(drop=True)
    frame['segment_id'] = np.arange(len(frame))
    return frame, dict(schema_version=1, corpus_sha256=digest.hexdigest(),
        embedding_model=os.getenv('EMBEDDING_MODEL', 'bge-m3'), lectures=lectures)

def ollama_url():
    return os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')

def model_digest():
    r = requests.get(f'{ollama_url()}/api/tags', timeout=15)
    r.raise_for_status()
    name = os.getenv('EMBEDDING_MODEL', 'bge-m3')
    name = name if ':' in name else name + ':latest'
    for model in r.json()['models']:
        if model['name'] == name:
            return model['digest']
    raise ValueError(f'Missing embedding model; run: ollama pull {name}')

def create_embeddings(texts):
    options = {}
    if os.getenv('EMBEDDING_NUM_GPU') is not None:
        options['num_gpu'] = int(os.environ['EMBEDDING_NUM_GPU'])
    r = requests.post(f'{ollama_url()}/api/embed', json={
        'model': os.getenv('EMBEDDING_MODEL', 'bge-m3'), 'input': texts,
        'truncate': False, 'keep_alive': '10m', 'options': options}, timeout=300)
    r.raise_for_status()
    return r.json()['embeddings']

def load_index():
    path = Path(os.getenv('INDEX_PATH', str(ROOT / 'lecture_index.joblib')))
    if not path.exists():
        raise FileNotFoundError('Missing lecture_index.joblib. Run: python preprocessjson.py')
    data = joblib.load(path)
    frame, matrix = data['df'], np.asarray(data['embedding_matrix'], dtype=np.float32)
    if not np.array_equal(frame.index.to_numpy(), np.arange(len(frame))):
        raise ValueError('Index needs a positional RangeIndex; rebuild it')
    if matrix.ndim != 2 or matrix.shape[0] != len(frame) or not len(frame):
        raise ValueError('Index row/embedding count mismatch')
    order = frame.sort_values(['lecture', 'start']).index.to_numpy()
    frame, matrix = frame.iloc[order].reset_index(drop=True), matrix[order]
    if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
        raise ValueError('Invalid embedding vectors')
    expected, meta = corpus()
    if data.get('metadata', {}).get('corpus_sha256') != meta['corpus_sha256']:
        raise ValueError('Stale or legacy index; run python preprocessjson.py --force')
    if data['metadata']['embedding_model'] != meta['embedding_model']:
        raise ValueError('Embedding model differs from index; rebuild it')
    columns = ['lecture', 'local_segment_id', 'start', 'end', 'text']
    # Compare values rather than pandas string-storage metadata across versions.
    if not np.array_equal(frame[columns].to_numpy(dtype=object), expected[columns].to_numpy(dtype=object)):
        raise ValueError('Index rows differ from corpus; rebuild it')
    return frame, matrix, data['metadata']

def expand_with_neighbors(frame, indices, window=1):
    expanded = set()
    for idx in indices:
        for i in range(max(0, idx-window), min(len(frame), idx+window+1)):
            if frame.iloc[i].lecture == frame.iloc[idx].lecture:
                expanded.add(i)
    return frame.iloc[sorted(expanded)]

def merge_segments(frame):
    merged = []
    for row in frame.itertuples():
        if merged and row.lecture == merged[-1]['lecture'] and row.start <= merged[-1]['end'] + 1:
            merged[-1]['end'] = max(merged[-1]['end'], row.end)
            merged[-1]['text'] += ' ' + row.text
        else:
            merged.append(dict(lecture=row.lecture, start=row.start, end=row.end, text=row.text))
    return merged

def retrieve(frame, matrix, question, window=1):
    start = perf_counter()
    vector = create_embeddings([question])[0]
    embedded = perf_counter()
    scores = cosine_similarity(matrix, [vector]).ravel()
    indices = np.argsort(-scores, kind='stable')[:5]
    searched = perf_counter()
    selected = expand_with_neighbors(frame, indices, window)
    contexts = merge_segments(selected) if scores[indices[0]] >= .25 else []
    finished = perf_counter()
    return dict(indices=indices, selected=selected, contexts=contexts, scores=scores,
        timings_ms=dict(embedding=(embedded-start)*1000, similarity=(searched-embedded)*1000,
        context=(finished-searched)*1000, retrieval=(finished-start)*1000))

def build_prompt(question, contexts):
    blocks = [f"[{i}] Lecture {c['lecture']} ({c['start']:.1f}s-{c['end']:.1f}s)\n{c['text']}"
              for i, c in enumerate(contexts, 1)]
    return ('Answer strictly from the lecture excerpts and cite their numbered references. '
            'If the answer is absent, say "I don\'t know based on the provided lectures."\n\n'
            f'Question: {question}\n\nLecture excerpts:\n' + '\n'.join(blocks) + '\n\nAnswer:')

def generation_options():
    return {'temperature': 0, 'seed': 42,
            'num_predict': int(os.getenv('LLM_NUM_PREDICT', '512')),
            'num_ctx': int(os.getenv('LLM_NUM_CTX', '8192'))}

def call_llm_details(prompt, options=None):
    settings = generation_options()
    if options:
        settings.update(options)
    started = perf_counter()
    r = requests.post(f'{ollama_url()}/api/generate', json={
        'model': os.getenv('LLM_MODEL', 'llama3.1:8b'), 'prompt': prompt, 'stream': False,
        'options': settings}, timeout=600)
    r.raise_for_status()
    payload = r.json()
    metrics = {k: payload.get(k) for k in ('model', 'done_reason', 'total_duration',
        'load_duration', 'prompt_eval_count', 'prompt_eval_cached_count',
        'prompt_eval_duration', 'eval_count', 'eval_duration')}
    metrics['wall_ms'] = (perf_counter()-started)*1000
    count, duration = metrics.get('eval_count'), metrics.get('eval_duration')
    metrics['output_tokens_per_second'] = count*1e9/duration if count and duration else None
    return dict(answer=payload['response'], metrics=metrics, options=settings)

def call_llm(prompt):
    return call_llm_details(prompt)['answer']
