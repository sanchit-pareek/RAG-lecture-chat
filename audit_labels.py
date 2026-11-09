"""Rescore saved rankings against reviewed passage labels without rerunning models."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from backend.rag import ROOT, corpus
from evaluate import relevant_row_indices, ranking_metrics, save

def main():
    source=ROOT/'evaluation/questions.json'
    reviewed_path=ROOT/'evaluation/questions_reviewed.json'
    original=json.loads(source.read_text(encoding='utf-8'))
    reviewed=json.loads(reviewed_path.read_text(encoding='utf-8'))
    saved=json.loads((ROOT/'evaluation/benchmark_results.json').read_text(encoding='utf-8'))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != reviewed['source_dataset_sha256'] or source_hash != saved['configuration']['dataset_sha256']:
        raise ValueError('Original dataset hash mismatch')
    frame,meta=corpus()
    if meta['corpus_sha256'] != saved['configuration']['metadata']['corpus_sha256']:
        raise ValueError('Corpus differs from saved results')
    originals={e['id']:e for e in original['examples']}
    by_query=defaultdict(list)
    for row in saved['results']:
        by_query[row['query_id']].append(row)
    if set(by_query)!=set(originals):
        raise ValueError('Missing or extra saved query IDs')
    lookup={(r.lecture,int(r.local_segment_id)):i for i,r in frame.iterrows()}
    rows=[]
    for example in reviewed['examples']:
        key=example['id']
        if example['question'] != originals[key]['question']:
            raise ValueError('Question changed; cannot reuse embeddings/rankings')
        runs=by_query[key]
        if any(r['question'] != example['question'] for r in runs):
            raise ValueError('Saved question text mismatch')
        predictions=[[(h['lecture'],h['segment']) for h in r['top5']] for r in runs]
        if any(p != predictions[0] for p in predictions):
            raise ValueError('Rankings varied across repeats; review separately before rescoring')
        indices=[lookup[p] for p in predictions[0]]
        rows.append(dict(query_id=key,question=example['question'],include_in_scoring=example['include_in_scoring'],
            original=ranking_metrics(indices,relevant_row_indices(frame,originals[key])),
            reviewed=ranking_metrics(indices,relevant_row_indices(frame,example)),
            note=example['review_note']))
    included=[r for r in rows if r['include_in_scoring']]
    summary=dict(queries_scored=len(included),excluded_queries=[r['query_id'] for r in rows if not r['include_in_scoring']],
        original_labels_same_subset={k:float(np.mean([r['original'][k] for r in included])) for k in included[0]['original']},
        reviewed_labels_same_subset={k:float(np.mean([r['reviewed'][k] for r in included])) for k in included[0]['reviewed']},
        changed_outcomes=[r['query_id'] for r in included if r['original']!=r['reviewed']])
    report=dict(status=reviewed['status'],limitations=reviewed['limitations'],summary=summary,
        source_dataset_sha256=source_hash,reviewed_dataset_sha256=hashlib.sha256(reviewed_path.read_bytes()).hexdigest(),
        corpus_sha256=meta['corpus_sha256'],results=rows)
    save(ROOT/'evaluation/label_audit_results.json',report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
