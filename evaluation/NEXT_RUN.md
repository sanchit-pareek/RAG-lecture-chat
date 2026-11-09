# Complete the three follow-ups

## Current status

The previous CPU profile covered one repeated question. No varied-question CPU or paired full-response results have yet been collected. On 9 November 2026 this agent session could not connect to the local API: Windows socket access was denied (10013). The running backend has therefore not been changed or verified by this session.

Existing relevance labels were authored/revised by an assistant after inspecting retrieval results. They remain exploratory. Independent validation requires another reviewer who has not inspected system rankings or answers; repeating the same assistant review does not establish independence.

## Run in your PowerShell terminal

With the project backend and Ollama already running:

```powershell
.\run-evaluation.ps1
```

This does not reinstall dependencies, rebuild the frontend, retranscribe, or rebuild the index. It:

1. Exports all 30 questions and proposed evidence into a source-only independent review packet, including the previously excluded question. Originals are preserved.
2. Tests CPU embeddings on all 30 original questions and requires exact top-five ranking agreement with the saved original benchmark before applying CPU placement. Agreement measures placement consistency, not accuracy.
3. Verifies ownership before restarting this project's backend with `EMBEDDING_NUM_GPU=0`, then records its runtime health and code fingerprint.
4. Runs an eight-question pilot: one excluded warmup and 16 paired full-response HTTP requests, with seeded randomized order. Requests have a 180-second read timeout; each completed answer is saved. Timeout does not cancel generation already running on the server. Do not immediately start another benchmark after a timeout.

The pilot deliberately includes definitions, numerical questions, derivations, and known retrieval misses. It is a varied diagnostic subset, not an unbiased estimate of all student traffic. Runtime depends on response lengths, model residency, and hardware; completion is not guaranteed within a fixed total time.

For all 29 currently included questions and three expanded-context passes:

```powershell
.\run-evaluation.ps1 -Full -Repeats 3
```

That larger run generates 116 measured answers plus one warmup and may take substantially longer. The first pass pairs both configurations; later passes measure expanded context only. Report first-pass and repeated-query timings separately because caching can affect repeats.

## Review before making claims

The timestamped run directory contains CPU ranking results, placement comparison, applied backend health, and `paired/api_results.json`. The latter records client HTTP time, component timings, answers, source excerpts, and generation counters. A completed run creates `paired/answer_review.json` with shuffled A/B answers. Score correctness, completeness, grounding, and citation support before inspecting `paired/review_key.json`. Higher source coverage alone is not an answer-quality gain.

Have an independent reviewer complete `independent-label-review.json` against full transcripts and document alternate valid passages and exclusions before using retrieval accuracy figures. The emitted ranking scores remain exploratory until those decisions are applied and rescored. Even independent review does not turn this source-derived, post-hoc set into a held-out benchmark.

No validated accuracy, representative full-response latency, or context-expansion quality improvement is claimed by the scripts or this document. Final numerical conclusions require completed runs and completed reviews.
