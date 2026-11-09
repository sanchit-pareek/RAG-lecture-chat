"""Export source evidence without rankings or system answers for an independent reviewer."""
import argparse
import hashlib
import json
from pathlib import Path
from backend.rag import ROOT, corpus
from evaluate import save, relevant_row_indices


def prepare(dataset_path, output):
    if output.exists():
        raise ValueError('Refusing to overwrite a review')
    raw = dataset_path.read_bytes()
    dataset = json.loads(raw)
    frame, metadata = corpus()
    examples = []
    for example in dataset['examples']:
        indices = relevant_row_indices(frame, example)
        evidence = frame.loc[sorted(indices), ['lecture', 'local_segment_id', 'start', 'end', 'text']]
        examples.append(dict(query_id=example['id'], question=example['question'],
            proposed_answer=example.get('expected_answer'),
            proposed_passages=example['relevant_passages'], source_evidence=evidence.to_dict('records'),
            previously_excluded=not example.get('include_in_scoring', True),
            answerable=None, labels_valid=None, corrected_passages=[], alternate_passages=[], notes=''))
    save(output, dict(status='pending_independent_review', reviewer='',
        dataset_sha256=hashlib.sha256(raw).hexdigest(), corpus_sha256=metadata['corpus_sha256'],
        instructions='Reviewer must be independent of question/label authors and must not inspect system rankings or answers. Read full transcripts to check answerability and alternate evidence; supplied passages are proposals, not gold labels. Complete every decision, document corrections, and identify yourself. This post-hoc dataset is not held-out, even after review. No accuracy claims are authorized by exporting this packet.',
        transcript_directory=str(ROOT/'transcripts'), examples=examples))
    print(f'Prepared {len(examples)} source-only review items: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'evaluation/questions_reviewed.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset, args.output)
