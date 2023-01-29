"""
accuracy score - even label ratio
"""

CLS_DICT = {'축정렬불량': 0,
            '회전체불평형': 1,
            '베어링불량': 2,
            '벨트느슨함': 3,
            '정상': 4,}


import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def load_csv(path):
    return pd.read_csv(path,skipinitialspace=True)

def evaluation_metric(label, prediction):
    return accuracy_score(label, prediction)

def evaluate(label, prediction):
    return evaluation_metric(label=label, prediction=prediction)

def load_result(path, pred=False):
    result = load_csv(path)
    if pred is False: # answer
        p_type_li = list(result['public'])
    else: # prediction
        p_type_li = None
    return list(result['file_name']), list(result['label']), p_type_li

def accuracy(answer_path, pred_path):

    a_id, answer, p_type_li = load_result(answer_path)
    p_id, pred, _ = load_result(pred_path, pred=True)
    
    a_id = sorted(a_id, key=str.lower)
    answer = sorted(answer, key=str.lower)
    p_id = sorted(p_id, key=str.lower)
    pred = sorted(pred, key=str.lower)

    assert a_id == p_id, f'Please match the order with the sample submission : {a_id}'
    assert len(a_id) == len(p_id), 'The number of predictions and answers are not the same'
    assert set(p_id) == set(a_id), 'The prediction ids and answer ids are not the same'

    pub_a_id, pub_answer, prv_a_id, prv_answer = [], [], [], []
    pub_p_id, pub_pred, prv_p_id, prv_pred = [], [], [], []
    
    for idx, t in enumerate(p_type_li):
        if t:
            pub_a_id.append(a_id[idx])
            pub_answer.append(answer[idx])
            pub_p_id.append(p_id[idx])
            pub_pred.append(pred[idx])            
            
        else:
            prv_a_id.append(a_id[idx])
            prv_answer.append(answer[idx])
            prv_p_id.append(p_id[idx])
            prv_pred.append(pred[idx]) 

    # sort
    pub_answer = pd.DataFrame({'file_name': pub_a_id, 'label': pub_answer}).sort_values('file_name', ignore_index=True)
    prv_answer = pd.DataFrame({'file_name': prv_a_id, 'label': prv_answer}).sort_values('file_name', ignore_index=True)
    pub_pred = pd.DataFrame({'file_name': pub_p_id, 'label': pub_pred}).sort_values('file_name', ignore_index=True)
    prv_pred = pd.DataFrame({'file_name': prv_p_id, 'label': prv_pred}).sort_values('file_name', ignore_index=True)

    
    score = evaluate(label=pub_answer['label'].map(CLS_DICT), prediction=pub_pred['label'].map(CLS_DICT))
    pScore = evaluate(label=prv_answer['label'].map(CLS_DICT), prediction=prv_pred['label'].map(CLS_DICT))
    
    return score, pScore

if __name__ == '__main__':
    answer = sys.argv[1]
    pred = sys.argv[2]

    try:
        import time
        start = time.time()
        score, pScore = accuracy(answer, pred)
        print(f'score={score},pScore = {pScore}')
        print(f'Elapsed Time: {time.time() - start}')

    except Exception as e:
        print(f'evaluation exception error: {e}', file=sys.stderr)
        sys.exit()
