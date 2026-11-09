# Generation profile findings

The CPU trial has now completed successfully for the repeated diagnostic question. See [CPU_EMBEDDING_RESULTS.md](CPU_EMBEDDING_RESULTS.md) for co-residency checks, timings and limits. The next-trial instructions below describe the experiment that produced those results.

The completed profile contains six generation requests for one question. This is a small local diagnostic, not representative end-to-end performance across the corpus.

## Observed pipeline behavior

| Run | Retrieval | Llama loading | Prompt processing | Output decoding | Diagnostic pipeline total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.45 s | 9.19 s | 0.43 s | 1.37 s | 15.48 s |
| 2 | 5.24 s | 8.79 s | 0.38 s | 1.37 s | 15.81 s |

Both pipeline runs show the same residency transition: Llama is loaded before the query, only BGE-M3 is listed after query embedding, and only Llama is listed after generation. The runtime therefore reloads models repeatedly during this workload. Llama loading dominates the generation phase. Embedding also takes 4.2–5.0 seconds here, versus the 72 ms median measured in the earlier warm embedding-only benchmark. The 205 ms warm-retrieval metric does not describe retrieval after switching away from Llama.

Llama generates about 19.7 output tokens/second in these pipeline runs. The 27-token answer decodes in about 1.37 seconds, much less than model loading. The snapshots show partial GPU residency for Llama, but do not establish whether eviction is caused by memory pressure, scheduler settings, or another runtime constraint. No server settings were changed or models force-unloaded.

## Context comparison

| Fixed-prompt request | Context | Load time | Cached input tokens | Wall time |
| --- | ---: | ---: | ---: | ---: |
| 1 | 8192 | 8.676 s | 0 | 10.57 s |
| 1 | 4096 | 8.067 s | 0 | 9.60 s |
| 2 | 4096 | 0.007 s | 280 | 1.66 s |
| 2 | 8192 | 7.387 s | 0 | 9.07 s |

The 4,096-context median of 5.63 seconds mixes a 9.60-second reload with a 1.66-second resident, cached-prompt request. Both 8,192-context requests reload the model. Comparing these medians as a context-size speedup would be misleading. The smallest 1.66-second result is not a full RAG query: it skips query embedding and reuses 280 of 281 prompt tokens. The earlier 67-second smoke result cannot be explained retrospectively because it lacks these counters.

## Next controlled diagnostic

The code now supports an explicit CPU-embedding trial using the same BGE-M3 model and existing index. The hypothesis is that keeping embeddings off the GPU may allow Llama to remain resident; this is unmeasured and is not the app default. Scheduler restrictions may still prevent co-residency. CPU query embeddings may be slower than warm GPU embeddings, so compare total pipeline time and model residency, not embedding time alone. Inspect top-five ranks for numerical placement differences.

Run from the normal PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe profile_generation.py --embedding-num-gpu 0 --contexts 8192 --pipeline-context 8192 --repeats 3 --output evaluation/generation_profile_cpu.json
```

This makes three fixed-prompt and three full-pipeline generation requests. It preserves the original profile. The 8,192 context in both phases matches the original pipeline, so this trial changes embedding placement while avoiding context-size switching. Compare residency and loading behavior. A separate 4,096-context trial can follow if needed. No reindex is required for this diagnostic because the embedding model and dimension are unchanged; verify rank consistency before adopting it.

Source files: `generation_profile.json`, `profile_generation.py`, `backend/rag.py`.
Ollama API references: [embedding runtime options](https://docs.ollama.com/api/embed), [generation timings](https://docs.ollama.com/api/generate), [model residency](https://docs.ollama.com/api/ps).
