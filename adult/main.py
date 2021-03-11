import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataset import preprocess_adult_data
from model import Net
from utils import train_dp, evaluate_dp
from utils import train_eo, evaluate_eo

def run_experiments(method='mixup', mode='dp', lam=0.5, num_exp=10):
    '''
    Retrain each model for 10 times and report the mean ap and dp.
    '''

    ap = []
    gap = []

    for i in range(num_exp):
        print('On experiment', i)
        # get train/test data
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed = i)

        # initialize model
        model = Net(input_size=len(X_train[0])).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # run experiments
        ap_val_epoch = []
        gap_val_epoch = []
        ap_test_epoch = []
        gap_test_epoch = []
        for j in tqdm(range(10)):
            if mode == 'dp':
                train_dp(model, criterion, optimizer, X_train, A_train, y_train, method, lam)
                ap_val, gap_val = evaluate_dp(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_dp(model, X_test, y_test, A_test)
            elif mode == 'eo':
                train_eo(model, criterion, optimizer, X_train, A_train, y_train, method, lam)
                ap_val, gap_val = evaluate_eo(model, X_val, y_val, A_val)
                ap_test, gap_test = evaluate_eo(model, X_test, y_test, A_test)

            if j > 0:
                ap_val_epoch.append(ap_val)
                ap_test_epoch.append(ap_test)
                gap_val_epoch.append(gap_val)
                gap_test_epoch.append(gap_test)

        # best model based on validation performance
        idx = gap_val_epoch.index(min(gap_val_epoch))
        gap.append(gap_test_epoch[idx])
        ap.append(ap_test_epoch[idx])


    print('--------AVG---------')
    print('Average Precision', np.mean(ap))
    print(mode + ' gap',  np.mean(gap))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adult Experiment')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/GapReg/erm')
    parser.add_argument('--mode', default='dp', type=str, help='dp/eo')
    parser.add_argument('--lam', default=0.5, type=float, help='Lambda for regularization')
    args = parser.parse_args()

    run_experiments(args.method, args.mode, args.lam)

