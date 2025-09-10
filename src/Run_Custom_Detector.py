# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

#  TriLinear Implementation based on Custom_AD

import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn, functional, optim
import random, argparse, time, os, logging
from sklearn.preprocessing import MinMaxScaler

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.models.base import BaseDetector
from TSB_AD.utils.utility import zscore
from TSB_AD.utils.torch_utility import EarlyStoppingTorch, get_gpu


def create_sequences(input, seq_length, horizon_size):
    xs, ys = [], []
    for i in range(len(input) - seq_length - horizon_size + 1):
        x = input[i:i + seq_length]
        if horizon_size == 1:
            y = input[i + seq_length]
        else:
            y = input[i + seq_length: i + seq_length + horizon_size]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def tricube_kernel(size):
    offset = (size - 1) / 2
    distances = torch.linspace(-offset, offset, steps=size) / offset  # normalize to [-1, 1]
    weights = (1 - torch.abs(distances) ** 3) ** 3
    weights = weights / weights.sum()
    return weights.to(dtype=torch.float32)

class TricubeSmoothing1D(nn.Module):
    def __init__(self, weights, stride=1):
        super(TricubeSmoothing1D, self).__init__()

        assert weights.ndim == 1, "Expected 1D tensor for weights"
        self.stride = stride

        weights = weights.view(1, 1, -1)
        self.register_buffer('weights', weights)

    def forward(self, x):
        B, T, C = x.shape
        kernel_size = self.weights.shape[-1]

        # Manual replicate padding
        pad_left = kernel_size // 2
        pad_right = kernel_size - 1 - pad_left
        front = x[:, 0:1, :].repeat(1, pad_left, 1)
        end = x[:, -1:, :].repeat(1, pad_right, 1)
        x = torch.cat([front, x, end], dim=1)


        # Prepare for conv1d
        x = x.permute(0, 2, 1)
        # weights = self.weights.to(x.device)
        out = functional.conv1d(x, self.weights.expand(C, 1, -1), stride=self.stride, groups=C)
        out = out.permute(0, 2, 1)
        return out


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, weights):
        super(series_decomp, self).__init__()
        self.tricube_smooth = TricubeSmoothing1D(weights, stride=1)

    def forward(self, x):
        out = self.tricube_smooth(x)
        residual = x - out
        return residual, out


class TriLinearModel(nn.Module):
    """
        In our implementation, we use a default setting of $N = \lceil 0.667 \times L \rceil$, where the factor $0.667$
        serves as a repetition ratio and the ceiling function ensures an integer window length.
        REF : https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html
    """
    def __init__(self, seq_length, output_size, kernel_frac=0.6666666666666666):
        super(TriLinearModel, self).__init__()

        # decomposition Kernel Size

        kernel_size = max(3, int(round(kernel_frac * seq_length)))
        self.weight = tricube_kernel(kernel_size)
        self.decomposition = series_decomp(self.weight)
        self.linear_seasonal = nn.Linear(seq_length, output_size)
        self.linear_trend = nn.Linear(seq_length, output_size)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        seasonal_output = self.linear_seasonal(seasonal_init)
        trend_output = self.linear_trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)



class Custom_AD(BaseDetector):

    def __init__(self, HP, normalize=True, epochs = 100, lr = 0.005):
        super(Custom_AD, self).__init__()
        self.__anomaly_score = None

        self.HP = HP
        self.normalize = normalize

        cuda = True

        self.cuda = cuda
        self.device = get_gpu(self.cuda)

        self.epochs = epochs
        self.lr = lr

        self.model = TriLinearModel(seq_length=seq_length, output_size=output_size, device=self.device).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.save_path = None
        self.early_stopping = EarlyStoppingTorch(save_path=self.save_path, patience=3)



    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, n_features = X.shape
        if self.normalize: X = zscore(X, axis=1, ddof=1)

        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)

            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)




        self.decision_scores_ = np.zeros(n_samples)
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        n_samples, n_features = X.shape
        decision_scores_ = np.zeros(n_samples)
        return decision_scores_


def run_Custom_AD_Unsupervised(data, HP):
    clf = Custom_AD(HP=HP)
    clf.fit(data)
    score = clf.decision_scores_
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

def run_Custom_AD_Semisupervised(data_train, data_test, HP):
    clf = Custom_AD(HP=HP)
    clf.fit(data_train)
    score = clf.decision_function(data_test)
    score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
    return score

if __name__ == '__main__':

    Start_T = time.time()
    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running Custom_AD')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='../Datasets/TSB-AD-U/')
    parser.add_argument('--AD_Name', type=str, default='Custom_AD')
    args = parser.parse_args()

    Custom_AD_HP = {
        'seq_length': [125],
        'horizon_size': [50]
    }

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()
    print('data: ', data.shape)
    print('label: ', label.shape)

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]

    start_time = time.time()

    output = run_Custom_AD_Semisupervised(data_train, data, **Custom_AD_HP)
    # output = run_Custom_AD_Unsupervised(data, **Custom_AD_HP)

    end_time = time.time()
    run_time = end_time - start_time

    pred = output > (np.mean(output)+3*np.std(output))
    evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=pred)
    print('Evaluation Result: ', evaluation_result)