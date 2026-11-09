"""Compare saved rankings without changing labels or treating placement as an accuracy gain."""
import argparse
import json
from pathlib import Path
from evaluate import save


def compare(baseline, candidate):
    for field in ('dataset_sha256', 'matrix_shape', 'top_k'):
        if baseline['configuration'][field] != candidate['configuration'][field]:
            raise ValueError(f'Cannot compare different {field}')
    for field in ('corpus_sha256','model_digest'):
        if baseline['configuration']['metadata'][field] != candidate['configuration']['metadata'][field]:
            raise ValueError(f'Cannot compare different {field}')
    def groups(report):
        result={}
        for row in report['results']:
            key=row['query_id']
            ranks=[(h['lecture'],h['segment']) for h in row['top5']]
            if key in result and (result[key]['ranks']!=ranks or result[key]['question']!=row['question']):
                raise ValueError(f'Rankings/text varied across repeats: {key}')
            result[key]=dict(ranks=ranks,question=row['question'],ranking=row['ranking'])
        return result
    old,new=groups(baseline),groups(candidate)
    if not old or set(old)!=set(new):
        raise ValueError('Query IDs differ or are empty')
    rows=[]
    for key in sorted(old):
        a,b=old[key],new[key]
        if a['question']!=b['question']:
            raise ValueError(f'Question text differs: {key}')
        rows.append(dict(query_id=key,exact_order_match=a['ranks']==b['ranks'],
            top1_match=a['ranks'][0]==b['ranks'][0],
            top5_set_overlap=len(set(a['ranks']) & set(b['ranks']))/len(a['ranks']),
            original=a['ranks'],candidate=b['ranks']))
    return dict(unique_questions=len(rows),all_top5_orders_match=all(r['exact_order_match'] for r in rows),
        matching_top1=sum(r['top1_match'] for r in rows),
        matching_top5_orders=sum(r['exact_order_match'] for r in rows),results=rows,
        limitation='This checks placement consistency on the existing source-derived benchmark, not independent answer accuracy.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('evaluation/benchmark_results.json'))
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-exact',action='store_true')
    a=p.parse_args()
    result=compare(json.loads(a.baseline.read_text(encoding='utf-8')),json.loads(a.candidate.read_text(encoding='utf-8')))
    save(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
    if a.require_exact and not result['all_top5_orders_match']:
        raise SystemExit('Rankings changed. Review the comparison before applying CPU placement.')
