# CPU embedding trial: measured results

## Result

Moving BGE-M3 query embeddings to the CPU allowed BGE-M3 and Llama 3.1 8B to remain resident together in every recorded pipeline snapshot. BGE-M3 reports zero VRAM use; Llama retains its original GPU allocation and 8,192-token context setting. Both models retain their original digests.

| Measurement | Original placement | CPU embeddings |
| --- | ---: | ---: |
| Full-pipeline diagnostic samples | 2 | 3 |
| Unique questions | 1 | 1 |
| Pipeline elapsed time | 15.48–15.81 s | 1.69–1.72 s |
| Median pipeline elapsed time | 15.65 s | 1.71 s |
| Retrieval elapsed time | 4.45–5.24 s | 0.245–0.290 s |
| Llama load time | 8.79–9.19 s | 0.005–0.017 s |
| Output decoding time | ~1.37 s | 1.33–1.37 s |
| Cached prompt tokens | 0 of 281 | 280 of 281 |

The residency snapshots and loading counters support the conclusion that this CPU placement avoids the observed repeated model loading for this question and runtime session. Loading is no longer the dominant cost in the CPU trial. The original and CPU prompts, delivered contexts, corpus hash, embedding model digest and answer text match. Top-five rankings are stable across the three CPU pipeline runs. The older profile did not record top-five scores, so an across-placement score comparison is unavailable; matching context/prompt text establishes the same delivered evidence for this question only.

## What these numbers support

This is a successful local diagnostic with a repeated prompt, not a general 1.7-second response-latency claim. The CPU requests benefit from a warm resident model and prompt cache. Avoid calling this a general 9x speedup: the samples are small, gathered sequentially, and differ in caching/residency state. Retaining the prompt cache is itself useful, but new questions will have different prompt processing and answer lengths.

The diagnostic pipeline timing also contains model-residency probe requests; it is not a measured FastAPI client HTTP response time. The original 205 ms warm retrieval benchmark remains a separate measurement from a different run/configuration.

No app default or running backend process was changed by this profiling command. CPU placement is currently an opt-in runtime setting (`EMBEDDING_NUM_GPU=0`), which a newly started backend can inherit. Do not infer that the already-running backend inherited the profiler's internal setting.

Before using the optimization in a resume performance claim, test varied questions with CPU placement, compare retrieval rankings against the saved baseline, and measure full API latency with the backend actually configured for CPU embeddings. Source completeness and citation quality also need separate evaluation. No index rebuild is needed merely to try placement with the same model; rank equivalence across the corpus has not yet been established.

## Resume wording supported now

- Built a lecture-grounded RAG assistant indexing 23,912 timestamped segments across 25 lectures (21.1 hours) with BGE-M3 embeddings and cosine-similarity retrieval.
- Developed a FastAPI and React pipeline with Llama 3.1 8B, timestamped sources, resumable indexing, and CPU/GPU placement profiling that identified and avoided repeated model loading in local tests.

For a compact version using the earlier retrieval measurement: “Built a RAG assistant over 25 lectures (21.1 hours; 23,912 segments), measuring 205 ms median warm retrieval latency across 90 local query runs.” Keep “warm retrieval” explicit, and keep the draft accuracy scores out until labels receive independent review.

## Evidence

- `generation_profile.json`: original six-request diagnostic, including two full-pipeline runs.
- `generation_profile_cpu.json`: six-request CPU diagnostic, including three full-pipeline runs.
- `cpu_profile_comparison.json`: hashes, invariant checks and machine-readable comparison.

All original results have been preserved.
