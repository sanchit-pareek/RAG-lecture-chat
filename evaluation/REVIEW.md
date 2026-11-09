# Label review and generation investigation

Reviewed 9 November 2026. The original question file and benchmark results are unchanged.

## Label audit

All 30 question/evidence pairs and their saved top-five results were read. `questions_reviewed.json` records expected answers, source passages, review notes and inclusion decisions. This is an assistant review performed after inspecting results. It is not independent or human validation, and alternate passages may remain unlabeled.

| Comparison on the same 29 questions | Original labels | Reviewed passage labels |
| --- | ---: | ---: |
| Hit@1 | 34.48% | 41.38% |
| Hit@5 | 65.52% (19/29) | 75.86% (22/29) |
| MRR@5 | 0.4420 | 0.5195 |

The original 30-question result remains 66.67% Hit@5. The 29-question comparison excludes q21, whose coefficient-identification task depends on a displayed parabola equation. There is no retriever improvement here: the saved rankings are identical; only the relevance labels and scored subset changed. Do not use the reviewed score as a held-out resume accuracy claim.

Notable findings:

- q01: rank 1 identifies the correct sample-space axiom, but the original label starts four segments too late. The support paragraph is segments 488–496 in lecture 1. Locating this paragraph is not proof that a three-segment expansion delivers its whole explanation.
- q02: the old support window includes generic radar framing beyond the two numerical facts. Trimming that unrelated tail removes a previously counted hit. The top-five excerpts discuss radar but do not deliver both numbers.
- q11: the old support window omits the sum and independence assumptions needed to understand the convolution result. The reviewed support passage includes the preceding derivation. Rewrite the question to state independence before a future independent benchmark.
- q14 and q30: alternate passages explain the requested concepts elsewhere in the same lecture. Added these to the labels.
- q15: extended the source window to include the actual conclusion that source identities are independent across arrivals.
- q23: the correct decomposition is variance plus squared bias. Added the preceding explanation that explicitly squares bias.

Remaining misses under the current reviewed labels: q02, q07, q10, q13, q20, q23, q24. These remain candidates for source inspection, alternate-passage labeling and chunking experiments. Short ASR fragments often match a topic while omitting the relation needed to answer the question.

Run the audit again without model inference:

```powershell
.\.venv\Scripts\python.exe audit_labels.py
```

The script validates corpus and dataset hashes, unchanged question text, query IDs, and identical rankings across repeats before rescoring. The evaluator now supports multiple relevant passages, including passages from different lectures.

The initial generation profile has now completed. See [GENERATION_FINDINGS.md](GENERATION_FINDINGS.md) for observed model reloading, measured component times, and the optional CPU-embedding trial. Earlier statements below about pending execution describe the initial agent attempt.

## Generation investigation

The existing smoke response records 67.34 seconds for generation but omits model loading, prompt processing and output decoding counters. No defensible cause can be assigned from that timing alone. GPU/CPU placement and model switching are hypotheses, not findings. The retrieved definition is also cut off mid-sentence after "every", and reference [2] does not state the definition; review source completeness and citations separately from answer plausibility.

`profile_generation.py` now records the official Ollama `load_duration`, `prompt_eval_duration`, `eval_duration`, input/output token counts, output tokens per second, and `/api/ps` model residency/VRAM metadata. It compares 8,192 versus 4,096 context settings using the exact same grounded prompt and fixed output cap, then repeats the actual embedding-before-generation pipeline. It does not force-unload models or change the application's default context setting.

From a normal PowerShell terminal, run:

```powershell
.\.venv\Scripts\python.exe profile_generation.py
```

This makes six generation requests by default and can take several minutes. Avoid simultaneous chat queries during the run. It saves progress after each completed request to `evaluation/generation_profile.json`. A first-observed request is not necessarily cold; repeated prompts can use the prompt cache; changing context settings may itself cause reloads. These effects are recorded rather than treated as a controlled production benchmark.

Interpretation after the run:

- Large load time: investigate model residency and repeated loads.
- Large prompt-evaluation time: investigate input length, prompt cache and runtime placement.
- Large decoding time with low output tokens/second: investigate inference placement and throughput.
- Residency changes between the embedding and generation snapshots: inspect whether the two models compete for available memory.
- Faster 4,096-context results: inspect answers, prompt token budget and loading/order effects before applying that setting or claiming an improvement.

Profiling was attempted from this agent session but local sockets remain blocked with Windows error 10013. No new generation measurements or speedup claims were produced. Original API defaults remain unchanged; `LLM_NUM_CTX` and `LLM_NUM_PREDICT` can now configure future deliberate trials.

Ollama primary references: [generation response counters](https://docs.ollama.com/api/generate), [running-model metadata](https://docs.ollama.com/api/ps).

## Resume bullets supported now

- Built a lecture-grounded RAG assistant indexing 23,912 timestamped segments across 25 lectures (21.1 hours) with BGE-M3 embeddings; measured 205 ms median retrieval latency over 90 local query runs.
- Developed a FastAPI and React question-answering pipeline integrating Llama 3.1 8B, timestamped citations, neighboring-segment context expansion, and a resumable embedding-index builder.

The latency refers to retrieval only, not full responses. Accuracy remains exploratory. Docker configuration exists, but successful Docker deployment has not been verified. Keep these bullets scoped accordingly.
