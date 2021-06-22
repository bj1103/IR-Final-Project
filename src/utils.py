import json
import numpy as np

def get_qids(folds_file: str, mode: str, test_folds: list, qrels: dict):
    tmp_qids = []
    with open(folds_file) as f_folds:
        folds = json.load(f_folds) 

    if mode == 'all':
        for i in range(len(folds)):
            tmp_qids += folds[i]
    elif mode == 'train':
        for i in range(len(folds)):
            if i not in test_folds:
                tmp_qids += folds[i]
    else:
        for i in test_folds:
            tmp_qids += folds[i]
    return [qid for qid in tmp_qids if qid in qrels.keys()]

def compute_MAP(prediction_file: str, qrels_file: str) -> float:
    qrels = json.load(open(qrels_file))
    prediction = json.load(open(prediction_file))
    MAP = 0
    q_cnt = 0
    for topic in prediction:
        if topic not in qrels:
            continue
        q_cnt += 1
        hit_num = 0
        for id, document in enumerate(prediction[topic]):
            try:
                if qrels[topic]['document'][document] > 0:
                    # print(id, document)
                    hit_num += 1
            except KeyError:
                continue
        doc_num = len(prediction[topic])
        prev_precision = 0
        now_hit_num = hit_num
        tmp_MAP = 0
        for index, document in enumerate(prediction[topic][::-1]):
            try:
                if qrels[topic]['document'][document] > 0:
                    prev_precision = max(prev_precision, now_hit_num / (doc_num - index))
                    now_hit_num -= 1
                    tmp_MAP += prev_precision
            except KeyError:
                continue
        MAP += tmp_MAP / qrels[topic]['relevant']
        print(f'{topic} with MAP: {tmp_MAP / qrels[topic]["relevant"]}')
        # input()
    MAP /= q_cnt
    return MAP

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prediction and Qrel files')
    parser.add_argument('prediction_file', type=str, help="Prediction file in json format")
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format ")

    argvs = parser.parse_args()
    score = compute_MAP(argvs.prediction_file, argvs.qrels_file)
    print(score)

