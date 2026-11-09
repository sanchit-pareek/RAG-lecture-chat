import unittest
from evaluation_suite import experiment_schedule, summarize, validate_health


class EvaluationProtocolTests(unittest.TestCase):
    def test_schedule_pairs_every_question_and_repeats_only_expanded(self):
        examples=[dict(id=f'q{i}') for i in range(8)]
        schedule=experiment_schedule(examples, 3, 42)
        self.assertEqual(schedule, experiment_schedule(examples, 3, 42))
        self.assertEqual(len(schedule), 32)
        for example in examples:
            self.assertEqual(sorted((r,w) for r,e,w in schedule if e==example),
                             [(0,0),(0,1),(1,1),(2,1)])

    def test_health_rejects_automatic_placement_and_stale_code(self):
        with self.assertRaises(ValueError):
            validate_health(dict(api_version='1.2.0',embedding_num_gpu='automatic'), 'corpus')
        with self.assertRaises(ValueError):
            validate_health(dict(api_version='1.2.0',embedding_num_gpu='0',
                                 corpus_sha256='corpus',code_sha256='stale'), 'corpus')

    def test_quality_comparison_rejects_changed_retrieval(self):
        def row(window, segment):
            return dict(query_id='q1',neighbor_window=window,repeat=0,api_ms=100,
                        response=dict(top5=[dict(lecture='a',segment=segment)]))
        with self.assertRaisesRegex(ValueError, 'rankings changed'):
            summarize([row(0,1),row(1,2)],1,1)


if __name__=='__main__':
    unittest.main()
