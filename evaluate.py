"""Corpus stats and reproducible retrieval/expansion evaluation."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from time import perf_counter
import numpy as np
import requests
from backend.rag import (ROOT, corpus, load_index, retrieve, expand_with_neighbors,
                         merge_segments, build_prompt, call_llm, model_digest, generation_options)

def save(path, data):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

def stats():
    frame, metadata = corpus()
    result = dict(lectures=len(metadata['lectures']), segments=len(frame),
        transcript_bytes=sum(x['bytes'] for x in metadata['lectures']),
        duration_hours=sum(x['duration'] for x in metadata['lectures'])/3600,
        average_segment_seconds=float((frame.end-frame.start).mean()),
        corpus_sha256=metadata['corpus_sha256'], embedding_shape=None)
    if (ROOT/'lecture_index.joblib').exists():
        _, matrix, _ = load_index()
        result['embedding_shape'] = list(matrix.shape)
    save(ROOT/'evaluation/corpus_stats.json', result)
    print(json.dumps(result, indent=2))

def percentiles(values):
    return dict(p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)), n=len(values))

def gpu_info():
    try:
        return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return 'Unavailable; record hardware manually'

def ranking_metrics(indices, relevant):
    ranks = [rank for rank, idx in enumerate(indices, 1) if int(idx) in relevant]
    return dict(hit_at_1=int(bool(ranks) and ranks[0] == 1), hit_at_5=int(bool(ranks)),
                reciprocal_rank=1/ranks[0] if ranks else 0)

def relevant_row_indices(frame, example):
    passages = example.get('relevant_passages') or [dict(
        lecture=example['relevant_lecture'], segments=example['relevant_segments'])]
    relevant = set()
    for passage in passages:
        ids = passage['segments']
        if not ids:
            raise ValueError(f'Empty labels: {example["id"]}')
        rows = frame[(frame.lecture == passage['lecture']) & frame.local_segment_id.isin(ids)]
        if len(rows) != len(set(ids)):
            raise ValueError(f'Invalid labels: {example["id"]}')
        relevant.update(rows.index)
    return relevant

def benchmark(args):
    frame, matrix, metadata = load_index()
    if metadata.get('model_digest') != model_digest():
        raise ValueError('Embedding model changed; rebuild the index')
    dataset = json.loads(args.dataset.read_text(encoding='utf-8'))
    examples = [e for e in dataset['examples'] if e.get('include_in_scoring', True)]
    if not examples or args.repeats < 1:
        raise ValueError('Need examples and positive repeats')
    # Labels are original, zero-based positions in each transcript JSON.
    labels = []
    for example in examples:
        labels.append(relevant_row_indices(frame, example))
    retrieve(frame, matrix, examples[0]['question'])  # warm-up excluded
    results = []
    for repeat in range(args.repeats):
        for number, (example, relevant) in enumerate(zip(examples, labels)):
            result = retrieve(frame, matrix, example['question'], window=0)
            scores = ranking_metrics(result['indices'], relevant)
            accepted = bool(result['contexts'])
            configurations = {}
            # Same top-5 and query embedding for both configurations.
            for window in (0, 1):
                start = perf_counter()
                selected = expand_with_neighbors(frame, result['indices'], window)
                contexts = merge_segments(selected) if accepted else []
                context_ms = (perf_counter()-start)*1000
                delivered = set(selected.index) if accepted else set()
                config = dict(context_ms=context_ms, context_segments=len(delivered),
                    context_words=sum(len(c['text'].split()) for c in contexts),
                    relevant_context_coverage=len(delivered & relevant)/len(relevant))
                config['retrieval_ms'] = (result['timings_ms']['embedding'] +
                    result['timings_ms']['similarity'] + context_ms)
                if args.generate:
                    prompt = build_prompt(example['question'], contexts)
                    started = perf_counter()
                    answer = call_llm(prompt) if accepted else "I don't know based on the provided lectures."
                    config.update(generation_ms=(perf_counter()-started)*1000, answer=answer,
                                  contexts=contexts)
                configurations[str(window)] = config
            row = dict(query_id=example['id'], repeat=repeat, question=example['question'],
                ranking=scores, accepted=accepted, timings_ms=result['timings_ms'],
                top5=[dict(lecture=frame.iloc[i].lecture, segment=int(frame.iloc[i].local_segment_id),
                           score=float(result['scores'][i])) for i in result['indices']],
                configurations=configurations)
            if args.api_url:
                start = perf_counter()
                response = requests.post(args.api_url.rstrip('/')+'/query',
                    json={'question': example['question']}, timeout=900)
                response.raise_for_status()
                row['api_ms'] = (perf_counter()-start)*1000
                row['api_response'] = response.json()
            results.append(row)
            print(f'Repeat {repeat+1}: {number+1}/{len(examples)}', flush=True)
            save(ROOT/'evaluation/benchmark_partial.json', results)
    summary = dict(dataset_status=dataset['status'], unique_queries=len(examples), repeats=args.repeats,
        ranking={k:float(np.mean([r['ranking'][k] for r in results]))
                 for k in ('hit_at_1','hit_at_5','reciprocal_rank')},
        latency_ms={k:percentiles([r['timings_ms'][k] for r in results])
                    for k in ('embedding','similarity','retrieval')},
        configurations={w:dict(mean_relevant_context_coverage=float(np.mean([
            r['configurations'][w]['relevant_context_coverage'] for r in results])),
            mean_context_words=float(np.mean([r['configurations'][w]['context_words'] for r in results])),
            context_ms=percentiles([r['configurations'][w]['context_ms'] for r in results])) for w in ('0','1')})
    for w in ('0','1'):
        summary['configurations'][w]['retrieval_ms'] = percentiles([
            r['configurations'][w]['retrieval_ms'] for r in results])
    if args.generate:
        for w in ('0','1'):
            summary['configurations'][w]['generation_ms'] = percentiles([
                r['configurations'][w]['generation_ms'] for r in results])
    if args.api_url:
        summary['api_ms'] = percentiles([r['api_ms'] for r in results])
    report = dict(summary=summary, configuration=dict(metadata=metadata,
        generation_model=os.getenv('LLM_MODEL','llama3.1:8b'), matrix_shape=list(matrix.shape),
        embedding_num_gpu=os.getenv('EMBEDDING_NUM_GPU','automatic'),
        top_k=5, threshold=.25, windows=[0,1], generation_options=generation_options(),
        dataset_sha256=__import__('hashlib').sha256(args.dataset.read_bytes()).hexdigest(),
        platform=platform.platform(), processor=platform.processor(), python=platform.python_version(),
        gpu=gpu_info(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','pandas','scikit-learn','requests','joblib')},
        ollama_tags=requests.get(__import__('backend.rag',fromlist=['ollama_url']).ollama_url()+'/api/tags',timeout=15).json()),
        results=results)
    save(args.output, report)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['stats','benchmark'])
    parser.add_argument('--dataset', type=Path, default=ROOT/'evaluation/questions.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--api-url')
    parser.add_argument('--embedding-num-gpu', type=int, choices=[0])
    parser.add_argument('--output', type=Path, default=ROOT/'evaluation/benchmark_results.json')
    args = parser.parse_args()
    if args.embedding_num_gpu is not None:
        os.environ['EMBEDDING_NUM_GPU'] = str(args.embedding_num_gpu)
    stats() if args.command == 'stats' else benchmark(args)
