"""Varied-question HTTP latency and paired context-expansion evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
from pathlib import Path
from time import perf_counter
import requests
from backend.rag import ROOT, corpus
from evaluate import save, percentiles, gpu_info, relevant_row_indices, ranking_metrics


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hash():
    return hashlib.sha256((ROOT/'backend/main.py').read_bytes()+(ROOT/'backend/rag.py').read_bytes()).hexdigest()


def validate_health(health, corpus_hash):
    if (health.get('api_version')!='1.2.0' or str(health.get('embedding_num_gpu'))!='0'
            or health.get('corpus_sha256')!=corpus_hash or health.get('code_sha256')!=code_hash()):
        raise ValueError('Backend is stale, uses different data, or is not configured for CPU embeddings. Run .\\run-evaluation.ps1.')


def request_query(api_url, question, window, timeout=180):
    started=perf_counter()
    response=requests.post(api_url.rstrip('/')+'/query',json=dict(question=question,neighbor_window=window),timeout=timeout)
    elapsed=(perf_counter()-started)*1000
    response.raise_for_status()
    payload=response.json()
    if payload.get('neighbor_window')!=window:
        raise ValueError('API did not honor the requested context configuration')
    return payload,elapsed


def experiment_schedule(examples, repeats, seed):
    rng=random.Random(seed)
    schedule=[]
    for repeat in range(repeats):
        shuffled=list(examples)
        rng.shuffle(shuffled)
        for example in shuffled:
            windows=[0,1] if repeat==0 else [1]
            rng.shuffle(windows)
            schedule.extend((repeat,example,window) for window in windows)
    return schedule


def summarize(rows, example_count, repeats):
    expanded=[r for r in rows if r['neighbor_window']==1]
    baseline=[r for r in rows if r['neighbor_window']==0]
    first_expanded=[r for r in expanded if r['repeat']==0]
    for row in baseline:
        partner=next(r for r in first_expanded if r['query_id']==row['query_id'])
        ranks=lambda r: [(h['lecture'],h['segment']) for h in r['response']['top5']]
        if ranks(row)!=ranks(partner):
            raise ValueError('Paired retrieval rankings changed; context-only comparison is invalid')
    return dict(unique_questions=example_count,repeats_expanded=repeats,
        expanded_api_ms=percentiles([r['api_ms'] for r in expanded]),
        expanded_first_pass_api_ms=percentiles([r['api_ms'] for r in first_expanded]),
        baseline_first_pass_api_ms=percentiles([r['api_ms'] for r in baseline]),
        expanded_generation_ms=percentiles([r['response']['timings_ms']['generation'] for r in expanded]),
        expanded_retrieval_ms=percentiles([r['response']['timings_ms']['retrieval'] for r in expanded]),
        ranking_first_pass={k:sum(r['ranking'][k] for r in first_expanded)/len(first_expanded)
                            for k in ('hit_at_1','hit_at_5','reciprocal_rank')},
        paired_context_coverage={str(w):sum(r['context_coverage'] for r in rows
            if r['repeat']==0 and r['neighbor_window']==w)/example_count for w in (0,1)},
        answer_quality_status='pending blinded review; coverage is not answer accuracy')


def make_review_packets(dataset, rows, output_dir, seed):
    """No arm names or timing/ranking metrics in the reviewer-facing answer packet."""
    rng=random.Random(seed+1)
    examples=[e for e in dataset['examples'] if e.get('include_in_scoring',True)]
    by_key={(r['query_id'],r['neighbor_window']):r for r in rows if r['repeat']==0}
    ratings,keys=[],[]
    for example in examples:
        order=[0,1]
        rng.shuffle(order)
        candidates=[]
        for letter,window in zip(('A','B'),order):
            row=by_key[(example['id'],window)]
            candidates.append(dict(candidate=letter,answer=row['response']['answer'],
                supplied_contexts=row['response']['contexts'],ratings=dict(correctness=None,
                    completeness=None,grounding=None,citation_support=None),notes=''))
            keys.append(dict(query_id=example['id'],candidate=letter,neighbor_window=window))
        ratings.append(dict(query_id=example['id'],question=example['question'],
            source_expected_answer=example['expected_answer'],source_passages=example['relevant_passages'],
            candidates=candidates))
    save(output_dir/'answer_review.json',dict(status='pending',reviewer='',
        instructions='Score 0=wrong/absent, 1=partial/mixed, 2=fully supported for each dimension. Correct abstention is appropriate when supplied evidence is insufficient. Judge correctness against source passages, grounding/citations against supplied excerpts. Do not open review_key.json before scoring. Record errors and ambiguous questions. This source-derived post-hoc dataset is not a held-out test.',
        examples=ratings))
    save(output_dir/'review_key.json',dict(assignments=keys))
    save(output_dir/'label_review.json',dict(status='pending',reviewer='',
        instructions='Independently read the linked full lecture transcripts and supplied passages. Confirm question answerability, expected answer, and support labels; add alternate passages across lectures where relevant. Leave no decision blank. No retrieval ranks or system answers are included. Human-reviewed source-derived labels remain post-hoc, not held-out.',
        examples=[dict(query_id=e['id'],question=e['question'],proposed_answer=e['expected_answer'],
            proposed_passages=e['relevant_passages'],transcript_paths=[str(ROOT/'transcripts'/(p['lecture']+'.json'))
                for p in e['relevant_passages']],decision=None,corrected_passages=[],notes='') for e in examples]))


def main(args):
    if args.repeats<1 or args.timeout<=0:
        raise ValueError('Repeats and timeout must be positive')
    if args.output_dir.exists():
        raise ValueError('Output directory already exists; choose a new run directory to preserve results')
    raw=args.dataset.read_bytes()
    dataset=json.loads(raw)
    examples=[e for e in dataset['examples'] if e.get('include_in_scoring',True)]
    if args.query_ids:
        requested=set(args.query_ids)
        if requested-set(e['id'] for e in examples):
            raise ValueError('Unknown or excluded query IDs')
        examples=[e for e in examples if e['id'] in requested]
        dataset=dict(dataset,examples=examples)
    if not examples:
        raise ValueError('No included questions')
    frame,metadata=corpus()
    labels={e['id']:relevant_row_indices(frame,e) for e in examples}
    if len(labels)!=len(examples):
        raise ValueError('Duplicate question IDs')
    lookup={(r.lecture,int(r.local_segment_id)):i for i,r in frame.iterrows()}
    response=requests.get(args.api_url.rstrip('/')+'/health',timeout=15)
    response.raise_for_status()
    health=response.json()
    validate_health(health,metadata['corpus_sha256'])
    report=dict(status='running',created_at=now(),dataset_status=dataset['status'],
        configuration=dict(dataset_sha256=hashlib.sha256(raw).hexdigest(),corpus_sha256=metadata['corpus_sha256'],
            health=health,seed=args.seed,query_ids=[e['id'] for e in examples],request_timeout_seconds=args.timeout,gpu=gpu_info(),platform=platform.platform(),
            warmup='One complete question excluded',protocol='Paired window 0/1 first pass in seeded random order; two more window-1 passes by default. Sequential HTTP requests, no forced model unload or cache clearing.'),
        results=[],errors=[],limitations='Post-hoc source-derived questions. Repeated questions and shared prompt prefixes can hit caches. Paired contexts differ in length. Report first-pass and all-repeat timings separately. No concurrent load test.')
    args.output_dir.mkdir(parents=True)
    save(args.output_dir/'api_results.json',report)
    try:
        warmup,wall=request_query(args.api_url,'What is a random variable?',1,args.timeout)
        report['warmup']=dict(api_ms=wall,response=warmup)
        schedule=experiment_schedule(examples,args.repeats,args.seed)
        for number,(repeat,example,window) in enumerate(schedule,1):
            payload,wall=request_query(args.api_url,example['question'],window,args.timeout)
            indices=[lookup[(h['lecture'],h['segment'])] for h in payload['top5']]
            delivered={lookup[(h['lecture'],h['segment'])] for h in payload['delivered_segments']}
            relevant=labels[example['id']]
            row=dict(query_id=example['id'],question=example['question'],repeat=repeat,
                neighbor_window=window,created_at=now(),api_ms=wall,response=payload,
                ranking=ranking_metrics(indices,relevant),context_coverage=len(delivered & relevant)/len(relevant))
            report['results'].append(row)
            save(args.output_dir/'api_results.json',report)
            print(f'{number}/{len(schedule)} {example["id"]} window={window} HTTP={wall/1000:.2f}s',flush=True)
        end=requests.get(args.api_url.rstrip('/')+'/health',timeout=15)
        end.raise_for_status()
        if end.json()!=health:
            raise ValueError('Backend configuration or process changed during the benchmark')
        report.update(status='complete',completed_at=now(),summary=summarize(report['results'],len(examples),args.repeats))
        make_review_packets(dataset,report['results'],args.output_dir,args.seed)
    except Exception as exc:
        report['status']='failed'
        report['errors'].append(dict(created_at=now(),error=str(exc),completed_queries=len(report['results'])))
        save(args.output_dir/'api_results.json',report)
        raise
    save(args.output_dir/'api_results.json',report)
    print(json.dumps(report['summary'],indent=2))
    print('Review packets:',args.output_dir/'answer_review.json',args.output_dir/'label_review.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-url',default='http://127.0.0.1:8000')
    parser.add_argument('--dataset',type=Path,default=ROOT/'evaluation/questions_reviewed.json')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--query-ids',nargs='+',help='Explicit varied-question pilot subset; omit for all included questions')
    parser.add_argument('--timeout',type=float,default=180,help='Per-request read timeout in seconds; failures are saved')
    parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args())
