import json
import numpy as np


def compute_MAP(prediction_file: str, qrels_file: str) -> float:
    qrels = json.load(open(qrels_file))
    prediction = json.load(open(prediction_file))
    MAP = 0
    for topic in prediction:
        hit_num = 0
        for document in prediction[topic]:
            if qrels[topic]['document'][document] == 1:
                hit_num += 1
        doc_num = len(prediction[topic])
        prev_precision = 0
        now_hit_num = hit_num
        for index, document in enumerate(prediction[topic][::-1]):
            if qrels[topic]['document'][document] == 1:
                prev_precision = max(prev_precision, now_hit_num / (doc_num - index))
                now_hit_num -= 1
                MAP += prev_precision
        MAP /= qrels[topic]['relevant']
    MAP /= len(prediction)
    return MAP

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prediction and Qrel files')
    parser.add_argument('prediction_file', type=str, help="Prediction file in json format")
    parser.add_argument('qrels_file', type=str, help="Qrel file in json format ")

    argvs = parser.parse_args()
    score = compute_MAP(argvs.prediction_file, argvs.qrels_file)
    print(score)
    
