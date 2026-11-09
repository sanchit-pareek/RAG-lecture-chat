import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
from backend import rag
from evaluate import ranking_metrics, relevant_row_indices
import preprocessjson

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            dict(lecture='a', local_segment_id=i, segment_id=i, start=float(i*2), end=float(i*2+2), text=f'a{i}')
            for i in range(3)] + [
            dict(lecture='b', local_segment_id=i, segment_id=i+3, start=float(i*2), end=float(i*2+2), text=f'b{i}')
            for i in range(3)])
        self.meta = dict(corpus_sha256='test', embedding_model='bge-m3')

    def test_neighbors_do_not_cross_lecture_boundary(self):
        self.assertEqual(rag.expand_with_neighbors(self.frame, [2], 1).index.tolist(), [1, 2])
        self.assertEqual(rag.expand_with_neighbors(self.frame, [3], 1).index.tolist(), [3, 4])

    def test_sort_keeps_vectors_attached_to_rows(self):
        order = [4, 0, 5, 1, 3, 2]
        vectors = np.arange(1, 13, dtype=np.float32).reshape(6, 2)
        saved_frame=self.frame.iloc[order].reset_index(drop=True)
        saved_frame['lecture']=saved_frame['lecture'].astype(object)
        saved_frame['text']=saved_frame['text'].astype(object)
        data = dict(df=saved_frame, embedding_matrix=vectors[order], metadata=self.meta)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            target = Path(directory)/'index.joblib'
            joblib.dump(data, target)
            with patch.dict(os.environ, INDEX_PATH=str(target)), patch.object(rag, 'corpus', return_value=(self.frame,self.meta)):
                loaded, matrix, _ = rag.load_index()
        self.assertEqual(loaded.text.tolist(), self.frame.text.tolist())
        np.testing.assert_array_equal(matrix, vectors)

    def test_retrieval_ranking_and_rejection(self):
        matrix = np.array([[1.,0.],[.5,.5],[0.,1.],[-1.,0.],[0.,-1.],[-.5,-.5]])
        with patch.object(rag,'create_embeddings',return_value=[[1.,0.]]):
            result = rag.retrieve(self.frame,matrix,'test')
        self.assertEqual(result['indices'][0],0)
        self.assertTrue(result['contexts'])
        with patch.object(rag,'create_embeddings',return_value=[[-1.,0.]]):
            result = rag.retrieve(self.frame,np.array([[1.,0.]]*6),'outside')
        self.assertEqual(result['contexts'],[])

    def test_metrics_use_first_relevant_rank(self):
        self.assertEqual(ranking_metrics([8,3,2,1,9],{2,1}),
                         dict(hit_at_1=0,hit_at_5=1,reciprocal_rank=1/3))
        self.assertEqual(ranking_metrics([8,3],{2})['reciprocal_rank'],0)

    def test_relevance_supports_alternate_lectures_and_validates_ids(self):
        example=dict(id='test',relevant_passages=[dict(lecture='a',segments=[0,1]),
            dict(lecture='b',segments=[2])])
        self.assertEqual(relevant_row_indices(self.frame,example),{0,1,5})
        example['relevant_passages'][1]['segments']=[20]
        with self.assertRaises(ValueError):
            relevant_row_indices(self.frame,example)

    def test_generation_preserves_ollama_telemetry_and_options(self):
        payload=dict(response='Answer',model='test',eval_count=20,eval_duration=2_000_000_000,
            load_duration=1_000_000_000,done_reason='stop')
        with patch.object(rag.requests,'post') as post:
            post.return_value.json.return_value=payload
            result=rag.call_llm_details('prompt',dict(num_ctx=4096))
            self.assertEqual(result['answer'],'Answer')
            self.assertEqual(result['metrics']['output_tokens_per_second'],10)
            self.assertEqual(result['metrics']['load_duration'],1_000_000_000)
            self.assertEqual(post.call_args.kwargs['json']['options']['num_ctx'],4096)
            self.assertIsNone(result['metrics']['prompt_eval_count'])

    def test_cpu_embedding_trial_is_explicit(self):
        with patch.dict(os.environ,{'EMBEDDING_NUM_GPU':'0'}), patch.object(rag.requests,'post') as post:
            post.return_value.json.return_value={'embeddings':[[1.,2.]]}
            self.assertEqual(rag.create_embeddings(['question']),[[1.,2.]])
            self.assertEqual(post.call_args.kwargs['json']['options'],{'num_gpu':0})

    def test_api_route_validation_and_answer(self):
        from fastapi import HTTPException
        from backend import main
        meta = dict(self.meta, model_digest='digest')
        with patch.object(main, 'df', self.frame), \
             patch.object(main, 'embedding_matrix', np.ones((6,2))), \
             patch.object(main, 'metadata', meta), \
             patch.object(rag, 'create_embeddings', return_value=[[1.,1.]]), \
             patch.object(main, 'call_llm_details', return_value=dict(answer='An answer [1].',metrics={},options={})):
            self.assertEqual(main.health()['segments'],6)
            with self.assertRaises(HTTPException) as error:
                main.query(main.QueryRequest(question='   '))
            self.assertEqual(error.exception.status_code,422)
            result=main.query(main.QueryRequest(question='What is independence?'))
            self.assertEqual(result['answer'],'An answer [1].')
            self.assertTrue(result['contexts'])
            self.assertGreaterEqual(result['timings_ms']['total'],0)

    def test_interrupted_index_resumes_cached_batches(self):
        calls = []
        def embed(texts):
            calls.append(texts)
            if len(calls)==2:
                raise RuntimeError('simulated interruption')
            return [[1.,2.] for _ in texts]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory, patch.object(preprocessjson,'ROOT',Path(directory)), \
             patch.object(preprocessjson,'corpus',return_value=(self.frame,self.meta.copy())), \
             patch.object(preprocessjson,'model_digest',return_value='digest'), \
             patch.object(preprocessjson,'create_embeddings',side_effect=embed):
            with self.assertRaises(RuntimeError):
                preprocessjson.build_index(2)
            self.assertFalse((Path(directory)/'lecture_index.joblib').exists())
            preprocessjson.build_index(2)
            result=joblib.load(Path(directory)/'lecture_index.joblib')
            self.assertEqual(result['embedding_matrix'].shape,(6,2))
            self.assertEqual(len(calls),4)  # one cached batch was not recomputed

if __name__=='__main__':
    unittest.main()
