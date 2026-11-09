# CPU placement and paired answer pilot

The saved run completed successfully with no recorded errors: 30 CPU retrieval queries and 16 measured full-response HTTP requests across eight distinct questions. One full-response warmup was excluded. `applied-health.json` and the paired run health metadata record CPU embedding placement (`embedding_num_gpu=0`), Llama 3.1 8B, an 8,192-token context, and a 512-token output cap. This confirms configuration at measurement time, not current process liveness.

## Measured latency

| Configuration | Samples | Median full-response HTTP time | p95 |
| --- | ---: | ---: | ---: |
| Raw top-five segments | 8 | 2.81 s | 10.26 s |
| Neighbor expansion | 8 | 3.58 s | 7.85 s |

The eight questions were a fixed diagnostic subset deliberately including known retrieval misses. These are sequential local pilot measurements after warmup, not a production latency SLA or an unbiased student-traffic benchmark. With only eight samples per condition, p95 is particularly unstable. Differences in output length, prompt caching, and residency can affect times; the table does not establish a causal latency improvement. Expanded median latency was higher in this pilot.

The separate 30-question CPU retrieval run measured 256.2 ms median and 320.9 ms p95 retrieval time. All 30 top-one results and exact top-five orders matched the saved automatic-placement benchmark. This supports placement consistency on these questions, not independently validated relevance or answer accuracy.

## Qualitative answer comparison

The assistant read the shuffled A/B packet before opening the assignment key. Context lengths make condition identity inferable, and this is not an independent human review. No numeric answer-quality score is claimed. Configuration names below are determined from the supplied contexts and checked against the key after recording judgments.

| Question | Raw segments | Expanded segments |
| --- | --- | --- |
| q01: sample-space probability | Wrong explanation based on nonzero individual probabilities; unsupported citation. | Correct certainty intuition, but omits the collectively exhaustive explanation. |
| q02: radar prior/detection | Abstains appropriately because the two numbers are absent. | Also abstains; needed numbers remain absent. |
| q07: functions preserve independence | Restates the claim without explaining why. | Still circular; missing information-based explanation. |
| q11: convolution | Incorrectly says convolution produces the joint PMF. | Appropriately abstains; relevant sum/independence explanation is missing. |
| q15: arrival source identities | Partial answer: one of two streams, without independence. | Explains binary source identity and independent coin flips, with supporting citations. |
| q20: normal tail symmetry | Appropriate abstention: needed evidence absent. | Same; no symmetry derivation retrieved. |
| q23: estimator MSE | Correct variance-plus-squared-bias answer, but its citation does not support it. | Incorrect decomposition into two variance terms; fails to supply squared bias. |
| q30: PMF value | Correct definition from outside the supplied fragments; citation does not establish it. | Correct definition, but the quoted sentence continues beyond the actual supplied excerpt. Citation remains unsupported. |

Expansion helps in some cases but does not reliably deliver complete evidence. Neither an overall quality improvement nor a grounded-answer accuracy percentage is justified. Short ASR fragments remain the central observed failure: topic matches omit explanatory relations, definitions, and sentence endings. In particular, mathematical correctness alone cannot be counted as grounded correctness.

## Still pending

Independent relevance validation remains unfinished: `independent-label-review.json` has no reviewer decisions. Existing Hit@k figures remain exploratory. The original `paired/answer_review.json` is preserved unmodified for a separate reviewer. Avoid using these eight cases as a held-out test after tuning chunking or prompts against them.

Evidence: `cpu-retrieval.json`, `placement-comparison.json`, `applied-health.json`, `paired/api_results.json`, and `paired/answer_review.json` in this directory.
