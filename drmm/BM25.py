from rank_bm25 import BM25Okapi
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import string
from utils import get_qids

def load_documents(cor: str, translator):
    docs_list = list()
    corpus = list()
    # with open(docs_file) as f_docs:
    #     print('Loading documents...')
    #     for doc in f_docs:
    #         doc = doc.strip()
    #         doc_path = Path(docs_dir) / doc
    #         with open(doc_path) as f_doc:
    #             corpus_tokens = f_doc.read().translate(translator).strip().lower().split()
    #             corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
    #         docs_list.append(doc)
    print('Loading documents...')
    for doc in cor:
        corpus_tokens = cor[doc].translate(translator).strip().lower().split()
        corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
        docs_list.append(doc)
    return docs_list, corpus
    
def compute_score(docs_file: str, topics_file: str, qrels_file: str, prediction_file: str, translator, rank_to_k=2000, mode='all', use_tag=["title", "description"]):
    with open(docs_file) as f:
        cor = json.load(f)
    with open(topics_file) as f:
        topics = json.load(f)
    with open(qrels_file) as f:
        qrels = json.load(f)

    docs_list, tokenized_corpus = load_documents(cor, translator)
    qids = get_qids(mode, qrels)

    bm25 = BM25Okapi(tokenized_corpus)
    rank_list = dict()
    for qid in tqdm(qids):
        query = ''
        for tag in use_tag:
            query += topics[qid][tag]
            query += ' '
        query_tokens = query.translate(translator).lower().split()
        tokenized_query = [tokens.strip() for tokens in query_tokens if tokens.strip() != '']
        doc_scores = bm25.get_scores(tokenized_query)
        top_k_ids = np.argsort(doc_scores)[::-1][:min(doc_scores.size, rank_to_k)]
        rank_list[qid] = list()
        for id in top_k_ids:
            rank_list[qid].append(docs_list[id])
        del doc_scores

    with open(prediction_file, 'w') as out:
        print(json.dumps(rank_list, indent=4), file=out)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Okapi score')
    parser.add_argument('docs_file', type=str, help="docs list in txt file")
    parser.add_argument('topics_file', type=str, help="Topic file in json format")
    parser.add_argument('qrels_file', type=str, help="Qrels file in json format")
    parser.add_argument('prediction_file', type=str, help="Output the prediction")
    parser.add_argument('--top_k', type=int, default=2000, help="Output the prediction")
    parser.add_argument('--mode', type=str, default='all', help="Output the prediction")
    argvs = parser.parse_args()

    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    compute_score(argvs.docs_file, argvs.topics_file, argvs.qrels_file, argvs.prediction_file, translator, rank_to_k=argvs.top_k, mode=argvs.mode)