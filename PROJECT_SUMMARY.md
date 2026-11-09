# RAG Teaching Assistant: project and resume summary

Updated 9 November 2026. Repository: [sanchit-pareek/RAG-lecture-chat](https://github.com/sanchit-pareek/RAG-lecture-chat).

## Recommended resume entry

**RAG Teaching Assistant** | Python, FastAPI, React, Ollama, BGE-M3, Llama 3.1 8B

- Built a lecture-grounded question-answering application with FastAPI and React, indexing **23,912 timestamped segments across 25 lectures (21.1 hours)** with BGE-M3 embeddings and Llama 3.1 8B.
- Engineered resumable embedding indexing, timestamped source retrieval, and CPU/GPU placement diagnostics; configured CPU query embeddings while preserving **identical top-five rankings across 30 test questions**.

Use these two bullets for the one-page Summer 2027 resume. They emphasize implementation, measurable scale, reliability, and performance investigation, relevant to software engineering, backend, full-stack, and AI/ML internships.

### Optional performance-focused second bullet

- Profiled model placement and configured CPU query embeddings, preserving top-five rankings on **30/30 questions** and measuring **256 ms median retrieval latency** across a 30-question local benchmark.

This latency is retrieval only, not full answer generation. Use this version if an execution metric fits the target role better than the indexing and source-display features. Link the repository for benchmark conditions.

### Short project description

A local RAG teaching assistant that turns timestamped lecture transcripts into a searchable question-answering application. It combines BGE-M3 semantic retrieval with Llama 3.1 8B generation, a FastAPI backend, a React interface, resumable indexing, source timestamps, and reproducible evaluation tools.

## Work completed

| Area | Delivered |
| --- | --- |
| Corpus verification | Counted 25 transcripts, 23,912 segments, 21.1163 hours from metadata, and a 23,912 x 1,024 embedding matrix. |
| Shared pipeline | Unified corpus loading, retrieval, prompts, and generation across API, CLI, and evaluation. |
| Index correctness | Kept transcript rows aligned with embedding rows during sorting; validated corpus fingerprints, model identity, vector shape, and values. |
| Resumable indexing | Cached fingerprinted batches and installed the completed index atomically. |
| Retrieval and sources | Top-five cosine search, same-lecture neighbor expansion, merged excerpts, and timestamped sources. Removed a post-expansion truncation that could discard evidence. |
| Backend and interface | Ran a model-backed FastAPI/React app, aligned frontend proxy paths, and used synchronous routes for blocking model requests. |
| Diagnostics | Recorded embedding, search, loading, prompt processing, decoding, token counts, and model residency. |
| CPU placement | Tested 30 varied questions; exact top-five orders matched the saved automatic-placement baseline for every question. Applied and verified CPU placement in the measured backend. |
| Full-response pilot | Completed 16 paired HTTP responses across eight questions, plus one excluded warmup, saving answers, sources, timings, and generation settings. |
| Quality analysis | Compared raw/expanded contexts and identified incomplete evidence, unsupported citations, appropriate abstention, and wrong answers. |
| Review preparation | Produced source-only independent relevance packets and shuffled A/B answer packets. Independent reviewer decisions remain pending. |
| Verification | Twelve regression tests passed; Python compilation and PowerShell syntax checks passed. |

## Verified measurements

| Measurement | Value | Scope |
| --- | ---: | --- |
| Lecture transcripts | 25 | Included corpus |
| Duration | 21.1163 hours | Sum of transcript metadata |
| Timestamped segments | 23,912 | Average duration about 3.15 s |
| Embedding dimensions | 1,024 | BGE-M3 index |
| CPU ranking agreement | 30/30 exact top-five orders | Consistency, not relevance accuracy |
| CPU embedding latency, median / p95 | 54 / 90 ms | 30-question run |
| CPU retrieval latency, median / p95 | 256 / 321 ms | Embedding, search, context preparation |
| Raw-context full HTTP latency, median / p95 | 2.81 / 10.26 s | Eight-question pilot |
| Expanded-context full HTTP latency, median / p95 | 3.58 / 7.85 s | Same questions, randomized paired order |

Recorded hardware included an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. Embeddings ran on CPU; Llama used Ollama's generation placement. Generation used temperature 0, seed 42, context 8,192, and output cap 512. These sequential local measurements do not establish concurrent throughput or production guarantees.

Earlier automatic-placement retrieval measured 205 ms median across 90 runs of 30 distinct questions. That is a different configuration/session: do not substitute it for the CPU result or present it as a CPU speedup. The older one-question placement diagnostic dropped from approximately 15.5–15.8 s to 1.69–1.72 s, but differed in caching and residency; it does not support a general 9x speedup.

## Answer comparison

Expansion improved the arrival-source explanation and replaced a wrong convolution answer with an appropriate abstention. It did not supply the missing radar numbers or normal-tail derivation, and it worsened the MSE answer. Some correct answers cited fragments that did not establish their result. Short ASR fragments often capture a topic while omitting the actual explanation.

This is an assistant qualitative review of eight diagnostic cases, not independent scoring. Expansion is an implemented feature, not a demonstrated overall quality improvement.

## Claims to keep out of the resume

- Validated retrieval/answer accuracy percentages: labels remain post-hoc and independently unreviewed. Hit@5 is passage localization, not answer correctness.
- General 9x speedup or 1.7-second response guarantees: those came from a repeated-question diagnostic with prompt caching.
- General 3.58-second latency guarantees: the pilot contains only eight distinct questions, including abstentions.
- Improved overall answer quality from expansion: results were mixed.
- Production readiness, concurrent scalability, or successful cloud/Docker deployment: unverified.
- Training or fine-tuning BGE-M3/Llama: the project integrates pretrained models.

## Interview explanation

The main engineering challenge was keeping retrieval correct and making local inference measurable with limited GPU memory. The implementation validates index identity and row alignment, resumes interrupted indexing, and exposes component timings. Placement profiling showed repeated model switching in the original diagnostic; CPU embeddings allowed both models to remain resident in recorded diagnostic snapshots. A broader CPU test preserved all top-five orders, and a separate HTTP pilot measured full responses. Evaluation also revealed that topic matching is insufficient when transcript fragments omit the explanation, making coherent passage chunking and independently reviewed labels the next priorities.

## Remaining work

1. Independently validate question answerability, source labels, alternate passages, and exclusions before publishing accuracy claims.
2. Compare coherent multi-sentence passages against segment indexing, with a separate held-out set for final evaluation.
3. Score correctness, completeness, grounding, and citation support separately.
4. Measure broader student questions before emphasizing full-response latency in the resume.

The project is ready for the resume with the recommended bullets above. These follow-ups should not delay internship applications.

## Evidence

- [Public results](evaluation/RESULTS.md) and [compact verified snapshot](evaluation/verified_results.json)
- [Corpus statistics](evaluation/corpus_stats.json)
- [CPU retrieval](evaluation/run-20261109-145055-867/cpu-retrieval.json) and [ranking comparison](evaluation/run-20261109-145055-867/placement-comparison.json)
- [Applied health](evaluation/run-20261109-145055-867/applied-health.json) and [paired HTTP results](evaluation/run-20261109-145055-867/paired/api_results.json)
- [Per-question findings](evaluation/run-20261109-145055-867/FINDINGS.md)
