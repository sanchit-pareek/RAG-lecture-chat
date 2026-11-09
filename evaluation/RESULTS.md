# Evaluation results

Updated 9 November 2026. These findings describe completed local measurements. Raw evidence is in `run-20261109-145055-867/`; [verified_results.json](verified_results.json) provides a compact snapshot with source-file hashes.

## Corpus and CPU placement

The corpus contains 25 transcripts, 23,912 timestamped segments, and 21.1163 hours from transcript metadata. The BGE-M3 matrix is 23,912 x 1,024.

CPU query embeddings preserved the exact top-five order for all 30 original benchmark questions against the saved automatic-placement baseline. Median CPU retrieval was 256.2 ms; p95 was 320.9 ms. Placement was applied and confirmed by the backend health recorded for the paired HTTP run.

## Full-response pilot

| Configuration | Distinct questions | Responses | HTTP median | HTTP p95 |
| --- | ---: | ---: | ---: | ---: |
| Raw segments | 8 | 8 | 2.81 s | 10.26 s |
| Adjacent-segment expansion | 8 | 8 | 3.58 s | 7.85 s |

One complete warmup was excluded; order was randomized within pairs. No errors were recorded. Questions span definitions, numerical facts, derivations, and known retrieval misses. The run used CPU query embeddings, Llama 3.1 8B, temperature 0, seed 42, 8,192 context tokens, and a 512-token output cap, on Windows with an RTX 4050 Laptop GPU (6 GB).

This small sequential pilot does not establish general latency, throughput, a causal speedup, or a stable p95. Caching, residency, and answer length affect timing.

## Answer quality

Assistant qualitative review found mixed effects. Expansion improved the arrival-source explanation and avoided a wrong convolution answer, but worsened the MSE answer. Some questions lacked sufficient evidence in either condition. Correct answers sometimes relied on information absent from the cited excerpt. No overall improvement is claimed.

See [the per-question comparison](run-20261109-145055-867/FINDINGS.md). The shuffled A/B packet was read before its key, but context lengths can reveal condition identity; this is not independent human scoring.

## Accuracy status

The original 30-question exploratory result was Hit@1 33.3%, Hit@5 66.7%, and MRR@5 0.4339. Post-hoc label review changed scoring without changing retrieval. Neither label set is independently validated, and passage localization does not measure answer correctness. Independent source-only review remains pending. These percentages should not be presented as validated resume achievements.

## Reproduction

With Ollama and the backend running, use `run-evaluation.ps1` for the pilot. Use `-Full -Repeats 3` for all 29 included questions and additional expanded passes. Each run saves completed responses in its own directory. See [README](../README.md) and [protocol details](../EVALUATION.md).

Twelve regression tests passed in the checked environment. Docker and Railway deployment remain unverified.
