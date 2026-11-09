"""Build a resumable, fingerprinted index without an interactive prompt."""
import argparse
import hashlib
import json
import joblib
import numpy as np
from backend.rag import ROOT, corpus, create_embeddings, model_digest, load_index

def build_index(batch_size=32, force=False):
    if batch_size < 1:
        raise ValueError('batch-size must be positive')
    frame, metadata = corpus()
    metadata['model_digest'] = model_digest()
    target = ROOT / 'lecture_index.joblib'
    if target.exists() and not force:
        _, _, existing = load_index()
        if existing.get('model_digest') != metadata['model_digest']:
            raise ValueError('Model digest changed; rebuild with --force')
        print(f'Valid index already exists: {target}', flush=True)
        return
    cache = ROOT / '.embedding-cache'
    cache.mkdir(exist_ok=True)
    matrices = []
    for start in range(0, len(frame), batch_size):
        texts = frame.iloc[start:start + batch_size].text.tolist()
        key = hashlib.sha256(json.dumps([metadata, texts], sort_keys=True).encode()).hexdigest()
        path = cache / f'{key}.joblib'
        if path.exists():
            vectors = joblib.load(path)
        else:
            vectors = np.asarray(create_embeddings(texts), dtype=np.float32)
            temporary = path.with_suffix('.tmp')
            joblib.dump(vectors, temporary)
            temporary.replace(path)
        if (vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all()
                or np.any(np.linalg.norm(vectors, axis=1) == 0)):
            raise ValueError('Invalid embedding response/cache')
        matrices.append(vectors)
        print(f'Indexed {min(start + batch_size, len(frame))}/{len(frame)} segments', flush=True)
    matrix = np.vstack(matrices)
    temporary = target.with_suffix('.tmp')
    joblib.dump(dict(df=frame, embedding_matrix=matrix, metadata=metadata), temporary)
    temporary.replace(target)
    print(f'Saved {target}: {matrix.shape}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    build_index(args.batch_size, args.force)
