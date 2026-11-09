"""Profile Ollama loading, prompt evaluation and decoding without changing API defaults."""
import argparse
import os
from datetime import datetime, timezone
from time import perf_counter
import numpy as np
import requests
from backend.rag import (ROOT, ollama_url, load_index, retrieve, build_prompt,
                         call_llm_details, model_digest)
from evaluate import save, gpu_info, percentiles

def resident():
    response=requests.get(ollama_url()+'/api/ps',timeout=15)
    response.raise_for_status()
    return response.json()

def main(args):
    if args.repeats < 1 or any(c < 1024 for c in args.contexts):
        raise ValueError('Use positive repeats and contexts >= 1024')
    if args.embedding_num_gpu is not None:
        os.environ['EMBEDDING_NUM_GPU']=str(args.embedding_num_gpu)
    frame,matrix,metadata=load_index()
    if model_digest()!=metadata['model_digest']:
        raise ValueError('Embedding model changed; rebuild index')
    rows=[]
    report=dict(created_at=datetime.now(timezone.utc).isoformat(), question=args.question,
        embedding_num_gpu=os.getenv('EMBEDDING_NUM_GPU','automatic'),
        pipeline_context=args.pipeline_context,
        gpu=gpu_info(), index_metadata=metadata, results=rows,
        limitations='First-observed requests are not guaranteed cold. Repeated identical prompts may use prompt cache. Alternating contexts may reload the model. No forced unloads; avoid other queries during profiling. This is a diagnostic, not a representative latency benchmark.')
    # First retrieve once. Reuse this exact grounded prompt throughout the context comparison.
    retrieved=retrieve(frame,matrix,args.question)
    if not retrieved['contexts']:
        raise ValueError('Question rejected by retrieval')
    prompt=build_prompt(args.question,retrieved['contexts'])
    report.update(prompt=prompt,contexts=retrieved['contexts'],initial_retrieval_ms=retrieved['timings_ms'])
    report['initial_top5']=[dict(lecture=frame.iloc[i].lecture,
        segment=int(frame.iloc[i].local_segment_id),score=float(retrieved['scores'][i]))
        for i in retrieved['indices']]
    save(args.output,report)
    required_tokens=None
    for repeat in range(args.repeats):
        # Alternate order to expose order effects instead of treating the first condition as cold.
        order=args.contexts if repeat%2==0 else list(reversed(args.contexts))
        for context in order:
            if required_tokens and context < required_tokens+512:
                rows.append(dict(repeat=repeat,num_ctx=context,skipped='Insufficient token budget based on first prompt evaluation'))
                save(args.output,report)
                continue
            before=resident()
            result=call_llm_details(prompt,dict(num_ctx=context,num_predict=512))
            required_tokens=max(required_tokens or 0,result['metrics'].get('prompt_eval_count') or 0)
            rows.append(dict(mode='fixed_prompt',repeat=repeat,num_ctx=context,
                resident_before=before,resident_after=resident(),**result))
            save(args.output,report)
            print('Fixed prompt',repeat+1,context,result['metrics'],flush=True)
    # Default context, fresh query embedding each time: detects model switching in the real pipeline.
    for repeat in range(args.repeats):
        before=resident()
        start=perf_counter()
        retrieved=retrieve(frame,matrix,args.question)
        after_embedding=resident()
        overrides=dict(num_ctx=args.pipeline_context) if args.pipeline_context else None
        result=call_llm_details(build_prompt(args.question,retrieved['contexts']),overrides)
        wall_ms=(perf_counter()-start)*1000
        rows.append(dict(mode='embedding_then_generation',repeat=repeat,
            resident_before=before,resident_after_embedding=after_embedding,resident_after=resident(),
            retrieval_ms=retrieved['timings_ms'],diagnostic_pipeline_ms=wall_ms,**result))
        rows[-1]['top5']=[dict(lecture=frame.iloc[i].lecture,
            segment=int(frame.iloc[i].local_segment_id),score=float(retrieved['scores'][i]))
            for i in retrieved['indices']]
        save(args.output,report)
        print('Pipeline',repeat+1,result['metrics'],flush=True)
    groups={}
    for row in rows:
        if 'metrics' not in row:
            continue
        key=row['mode']+':'+str(row['options']['num_ctx'])
        groups.setdefault(key,[]).append(row)
    report['summary']={key:dict(wall_ms=percentiles([r['metrics']['wall_ms'] for r in samples]),
        mean_load_ms=float(np.mean([(r['metrics'].get('load_duration') or 0)/1e6 for r in samples])),
        mean_prompt_ms=float(np.mean([(r['metrics'].get('prompt_eval_duration') or 0)/1e6 for r in samples])),
        mean_decode_ms=float(np.mean([(r['metrics'].get('eval_duration') or 0)/1e6 for r in samples])))
        for key,samples in groups.items()}
    save(args.output,report)
    print(report['summary'])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question',default='What is a random variable?')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--contexts',nargs='+',type=int,default=[8192,4096])
    parser.add_argument('--embedding-num-gpu',type=int,choices=[0],
        help='Optional CPU embedding trial; default keeps automatic placement')
    parser.add_argument('--pipeline-context',type=int,choices=[2048,4096,8192],
        help='Context setting for the embedding-then-generation phase only')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'evaluation/generation_profile.json')
    main(parser.parse_args())
