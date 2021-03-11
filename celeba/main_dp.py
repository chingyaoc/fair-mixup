import os
import argparse
import pandas as pd
import numpy as np
from numpy.random import beta
from tqdm import tqdm
import pickle
from pprint import pprint

import torch
import torch.nn as nn
from torch import optim

from model import *
from utils import *


def test_model(model, model_linear, dataloader, dataloader_0, dataloader_1):
    model.eval()
    model_linear.eval()

    ap, _ = evaluate(model, model_linear, dataloader)
    ap_0, score_0 = evaluate(model, model_linear, dataloader_0)
    ap_1, score_1 = evaluate(model, model_linear, dataloader_1)
    gap = abs(score_0 - score_1)
    pprint("AP: {:.4f} DP Gap: {:.4f}".format(ap, gap))



def fit_model(epochs, dataloader, dataloader_0, dataloader_1, mode='mixup_smooth', lam=100):
    pprint("Epoch: {}".format(epochs))

    len_dataloader = min(len(dataloader), len(dataloader_0), len(dataloader_1))
    len_dataloader = int(len_dataloader)
    data_iter = iter(dataloader)
    data_iter_0 = iter(dataloader_0)
    data_iter_1 = iter(dataloader_1)

    model.train()    
    model_linear.train()

    for it in tqdm(range(len_dataloader)):
        inputs_0, target_0 = data_iter_0.next()
        inputs_1, target_1 = data_iter_1.next()
        inputs_0, target_0 = inputs_0.cuda(), target_0.float().cuda()
        inputs_1, target_1 = inputs_1.cuda(), target_1.float().cuda()

        # supervised loss
        inputs = torch.cat((inputs_0, inputs_1), 0)
        target = torch.cat((target_0, target_1), 0)
        feat = model(inputs)
        ops = model_linear(feat)
        loss_sup = criterion(ops[:,0], target)

        if mode == 'GapReg':
            feat = model(inputs_0)
            ops_0 = model_linear(feat)
            feat = model(inputs_1)
            ops_1 = model_linear(feat)

            loss_gap = torch.abs(ops_0.mean() - ops_1.mean())
            loss = loss_sup + lam*loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} | Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif mode == 'mixup':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Input Mixup
            inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            feat = model(inputs_mix)
            ops = model_linear(feat).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup {:.7f}".format(loss_sup, loss_grad))

        elif mode == 'mixup_manifold':
            alpha = 1
            gamma = beta(alpha, alpha)

            # Manifold Mixup
            feat_0 = model(inputs_0)
            feat_1 = model(inputs_1)
            inputs_mix = feat_0 * gamma + feat_1 * (1 - gamma)
            inputs_mix = inputs_mix.requires_grad_(True)
            ops = model_linear(inputs_mix).sum()

            # Smoothness Regularization
            gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
            x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
            grad_inn = (gradx * x_d).sum(1).view(-1)
            loss_grad = torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup Manifold {:.7f}".format(loss_sup, loss_grad))

        else:
            loss = loss_sup
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f}".format(loss_sup))

        optimizer.zero_grad()
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_linear.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CelebA Experiment')
    parser.add_argument('--method', default='mixup', type=str, help='mixup/mixup_manifold/GapReg/erm')
    parser.add_argument('--lam', default=20, type=float, help='Lambda for regularization')
    parser.add_argument('--target_id', default=2, type=int, help='2:attractive/31:smile/33:wavy hair')
    args = parser.parse_args()

    # Load Celeb dataset
    target_id = args.target_id
    with open('celeba/data_frame.pickle', 'rb') as handle:
        df = pickle.load(handle)
    train_df = df['train']
    valid_df = df['val']
    test_df = df['test']

    train_dataloader = get_loader(train_df, 'celeba/split/train/', target_id, 64)
    valid_dataloader = get_loader(valid_df, 'celeba/split/val/', target_id, 64)
    test_dataloader = get_loader(test_df, 'celeba/split/test/', target_id, 64)

    train_dataloader_0 = get_loader(train_df, 'celeba/split/train/', target_id, 64, gender = '0')
    train_dataloader_1 = get_loader(train_df, 'celeba/split/train/', target_id, 64, gender = '1')
    valid_dataloader_0 = get_loader(valid_df, 'celeba/split/val/', target_id, 64, gender = '0')
    valid_dataloader_1 = get_loader(valid_df, 'celeba/split/val/', target_id, 64, gender = '1')
    test_dataloader_0 = get_loader(test_df, 'celeba/split/test/', target_id, 64, gender = '0')
    test_dataloader_1 = get_loader(test_df, 'celeba/split/test/', target_id, 64, gender = '1')

    # model
    model = ResNet18_Encoder(pretrained=False).cuda()
    model = nn.DataParallel(model)
    model_linear = LinearModel().cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr = 1e-3)

    for i in range(1, 100):
        fit_model(i, train_dataloader, train_dataloader_0, train_dataloader_1, args.method, args.lam)
        print('val:')
        test_model(model, model_linear, valid_dataloader, valid_dataloader_0, valid_dataloader_1)
        print('test:')
        test_model(model, model_linear, test_dataloader, test_dataloader_0, test_dataloader_1)

