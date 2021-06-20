from rank_bm25 import BM25Okapi
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import string
# word = word.translate().strip().split()
def load_documents(docs_file: str, docs_dir: str, translator):
    docs_list = list()
    corpus = list()
    with open(docs_file) as f_docs:
        print('Loading documents...')
        for doc in f_docs:
            doc = doc.strip()
            doc_path = Path(docs_dir) / doc
            with open(doc_path) as f_doc:
                corpus_tokens = f_doc.read().translate(translator).strip().lower().split()
                corpus.append([token.strip() for token in corpus_tokens if token.strip() != ''])
            docs_list.append(doc)
    return docs_list, corpus
    
def compute_score(docs_file: str, topics_file: str, docs_dir: str, prediction_file: str, translator, rank_to_k=200, use_tag=["title", "description"]):
    docs_list, tokenized_corpus = load_documents(docs_file, docs_dir, translator)

    bm25 = BM25Okapi(tokenized_corpus)
    topics = json.load(open(topics_file))
    rank_list = dict()
    for qid in tqdm(topics):
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
    parser.add_argument('docs_dir', type=str, help="Doc dir")
    parser.add_argument('prediction_file', type=str, help="Output the prediction")
    argvs = parser.parse_args()

    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    compute_score(argvs.docs_file, argvs.topics_file, argvs.docs_dir, argvs.prediction_file, translator)