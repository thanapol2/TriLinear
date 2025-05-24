import numpy as np

def txt_to_list(csv_name='temp.txt', is_string=True):
    with open(csv_name, 'r') as f:
        if is_string:
            list_txt = [line.strip() for line in f]
        else:
            list_txt = [float(line.strip()) for line in f]
    return list_txt


def create_sequences(data, seq_length, horizon_size):
    xs, ys = [], []
    for i in range(len(data) - seq_length - horizon_size + 1):
        x = data[i:i+seq_length]
        if horizon_size == 1:
            y = data[i + seq_length]
        else:
            y = data[i+seq_length: i+seq_length+horizon_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def anomaly_scores(preds, vals, is_minmax = True):
    np_anomaly_scores = []
    for pred, val in zip(preds, vals):
        scores = np.linalg.norm(val - pred)
        np_anomaly_scores.append(scores)

    np_anomaly_scores = np.array(np_anomaly_scores)

    if is_minmax:
        min_val = np.min(np_anomaly_scores)
        max_val = np.max(np_anomaly_scores)
        if max_val - min_val > 0:  # Avoid division by zero
            np_anomaly_scores = (np_anomaly_scores - min_val) / (max_val - min_val)
        else:
            np_anomaly_scores = np.zeros_like(np_anomaly_scores)

    return np.array(np_anomaly_scores)