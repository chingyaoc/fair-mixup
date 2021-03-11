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


def test_model(model, model_linear, dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11):
    model.eval()
    model_linear.eval()

    ap, _ = evaluate(model, model_linear, dataloader)

    _, score_00 = evaluate(model, model_linear, dataloader_00)
    _, score_01 = evaluate(model, model_linear, dataloader_01)
    _, score_10 = evaluate(model, model_linear, dataloader_10)
    _, score_11 = evaluate(model, model_linear, dataloader_11)

    gap_0 = abs(score_00 - score_10)
    gap_1 = abs(score_01 - score_11)
    gap = gap_0 + gap_1

    pprint("AP: {:.4f} EO Gap: {:.4f}".format(ap, gap))


def fit_model(epochs, model, dataloader, dataloader_00, dataloader_01, dataloader_10, dataloader_11, mode='mixup', lam=30): 

    len_dataloader = min(len(dataloader), len(dataloader_00), len(dataloader_01), len(dataloader_10), len(dataloader_11))
    data_iter = iter(dataloader)
    data_iter_00 = iter(dataloader_00)
    data_iter_01 = iter(dataloader_01)
    data_iter_10 = iter(dataloader_10)
    data_iter_11 = iter(dataloader_11)

    model.train()    
    model_linear.train()

    for it in tqdm(range(len_dataloader)):
        inputs_00, target_00 = data_iter_00.next()
        inputs_01, target_01 = data_iter_01.next()
        inputs_10, target_10 = data_iter_10.next()
        inputs_11, target_11 = data_iter_11.next()

        inputs_00, inputs_01 = inputs_00.cuda(), inputs_01.cuda()
        inputs_10, inputs_11 = inputs_10.cuda(), inputs_11.cuda()
        target_00, target_01 = target_00.float().cuda(), target_01.float().cuda()
        target_10, target_11 = target_10.float().cuda(), target_11.float().cuda()

        inputs_0_ = [inputs_00, inputs_01]
        inputs_1_ = [inputs_10, inputs_11]

        inputs = torch.cat((inputs_00, inputs_01, inputs_10, inputs_11), 0)
        target = torch.cat((target_00, target_01, target_10, target_11), 0)
        feat = model(inputs)
        ops = model_linear(feat)

        loss_sup = criterion(ops[:,0], target)

        if mode == 'GapReg':
            loss_gap = 0
            for g in range(2):
                inputs_0 = inputs_0_[g]
                inputs_1 = inputs_1_[g]
                feat = model(inputs_0)
                ops_0 = model_linear(feat)
                feat = model(inputs_1)
                ops_1 = model_linear(feat)

                loss_gap += torch.abs(ops_0.mean() - ops_1.mean())

            loss = loss_sup + lam*loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif mode == 'mixup':
            alpha = 1
            loss_grad = 0
            for g in range(2):
                inputs_0 = inputs_0_[g]
                inputs_1 = inputs_1_[g]
                gamma = beta(alpha, alpha)
                inputs_mix = inputs_0 * gamma + inputs_1 * (1 - gamma)
                inputs_mix = inputs_mix.requires_grad_(True)

                feat = model(inputs_mix)
                ops = model_linear(feat)
                ops = ops.sum()

                gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
                x_d = (inputs_1 - inputs_0).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_grad += torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup: {:.7f}".format(loss_sup, loss_grad))

        elif mode == 'mixup_manifold':
            alpha = 1
            loss_grad = 0
            for g in range(2):
                inputs_0 = inputs_0_[g]
                inputs_1 = inputs_1_[g]

                gamma = beta(alpha, alpha)
                feat_0 = model(inputs_0)
                feat_1 = model(inputs_1)
                inputs_mix = feat_0 * gamma + feat_1 * (1 - gamma)
                inputs_mix = inputs_mix.requires_grad_(True)

                ops = model_linear(inputs_mix)
                ops = ops.sum()

                gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
                x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_grad += torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup Manifold: {:.7f}".format(loss_sup, loss_grad))
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
    parser.add_argument('--lam', default=30, type=float, help='Lambda for regularization')
    parser.add_argument('--target_id', default=2, type=int, help='2:attractive/31:smile/33:wavy hair')
    args = parser.parse_args()

    # Load Celeb dataset
    target_id = args.target_id
    with open('celeba/data_frame.pickle', 'rb') as handle:
        df = pickle.load(handle)
    train_df = df['train']
    valid_df = df['val']
    test_df = df['test']

    # data_loader
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataloader = get_loader(train_df, 'celeba/split/train/', target_id, 64)
    valid_dataloader = get_loader(valid_df, 'celeba/split/val/', target_id, 64)
    test_dataloader = get_loader(test_df, 'celeba/split/test/', target_id, 64)

    train_dataloader_00 = get_loader(train_df, 'celeba/split/train/', target_id, 32, gender='0', target='0')
    train_dataloader_01 = get_loader(train_df, 'celeba/split/train/', target_id, 32, gender='0', target='1')
    train_dataloader_10 = get_loader(train_df, 'celeba/split/train/', target_id, 32, gender='1', target='0')
    train_dataloader_11 = get_loader(train_df, 'celeba/split/train/', target_id, 32, gender='1', target='1')

    valid_dataloader_00 = get_loader(valid_df, 'celeba/split/val/', target_id, 32, gender = '0', target='0') 
    valid_dataloader_01 = get_loader(valid_df, 'celeba/split/val/', target_id, 32, gender = '0', target='1')
    valid_dataloader_10 = get_loader(valid_df, 'celeba/split/val/', target_id, 32, gender = '1', target='0')
    valid_dataloader_11 = get_loader(valid_df, 'celeba/split/val/', target_id, 32, gender = '1', target='1')
 
    test_dataloader_00 = get_loader(test_df, 'celeba/split/test/', target_id, 32, gender = '0', target='0')
    test_dataloader_01 = get_loader(test_df, 'celeba/split/test/', target_id, 32, gender = '0', target='1')
    test_dataloader_10 = get_loader(test_df, 'celeba/split/test/', target_id, 32, gender = '1', target='0')
    test_dataloader_11 = get_loader(test_df, 'celeba/split/test/', target_id, 32, gender = '1', target='1')

    # model
    model = ResNet18_Encoder(pretrained=False).cuda()
    model = nn.DataParallel(model)
    model_linear = LinearModel().cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    optimizer_linear = optim.Adam(model_linear.parameters(), lr = 1e-3)

    for i in range(1, 250):
        fit_model(i, model, train_dataloader, train_dataloader_00, train_dataloader_01, train_dataloader_10, train_dataloader_11, args.method, args.lam)
        if i % 5 == 0:
            pprint("Epoch: {}".format(i))
            print('val:')
            test_model(model, model_linear, valid_dataloader, valid_dataloader_00, valid_dataloader_01, valid_dataloader_10, valid_dataloader_11)
            print('test:')
            test_model(model, model_linear, test_dataloader, test_dataloader_00, test_dataloader_01, test_dataloader_10, test_dataloader_11)
