import numpy as np
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision + recall)
    except Exception as e:
        return repr(e)
