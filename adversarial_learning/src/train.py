
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import os
from model import RoleModel, Discriminator
from translator import exec_translate
from util import *
import argparse
import logging
import shutil
import torch.nn.functional as F


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test_label", action="store_true", \
                        help="True if test labels are available for calculating test accuracy")

    parser.add_argument("--is_val_label", action="store_true", \
                        help="True if validation labels are available for calculating validation accuracy")

    parser.add_argument("--param_dir", default=None, \
                        help="preserve features and predicted labels")

    parser.add_argument("--epoch", default=3000, type=int, \
                        help="How many times models iterate")

    parser.add_argument("--dr_rate", default=0.5, type=float, \
                        help="Drop out rate should be set from 0 to 1")

    parser.add_argument("--lambda_", default=10, type=float, \
                        help="Balance coefficient")

    parser.add_argument("--wd", default=1e-4, type=float, \
                        help="weight decay rate")

    parser.add_argument("--task_lr", default=1e-4, type=float, \
                        help="learning rate for Role-Model")

    parser.add_argument("--disc_lr", default=1e-4, type=float, \
                        help="learning rate for Discriminator")

    args = parser.parse_args()
    return args


def save_params(param_dir, hparam_suffix, after_train_features, after_test_features, test_outputs):
    if not os.path.exists(param_dir):
        os.mkdir(param_dir)
    np.save(param_dir+"/train_feature_{}.npy".format(hparam_suffix), after_train_features)
    np.save(param_dir+"/test_feature_{}.npy".format(hparam_suffix), after_test_features)
    np.save(param_dir+"/test_outputs_{}.npy".format(hparam_suffix), test_outputs)


def train(D, D_criterion, D_labels, D_optimizer, R, R_criterion, y_train, R_optimizer, lambda_, X_train):
    # Discriminator optimization
    D.train()
    detached_weight = R.embed.weight.detach()
    D_outputs = D(detached_weight)

    D_loss = lambda_ * D_criterion(D_outputs, D_labels)

    D_optimizer.zero_grad()
    D_loss.backward(retain_graph=True)
    D_optimizer.step()

    # Role-model optimization
    R.train()
    R_outputs = R(X_train)

    move_weight = R.embed.weight
    D_outputs = D(move_weight)

    R_loss = R_criterion(R_outputs, y_train.long()) \
                - lambda_ * D_criterion(D_outputs, D_labels)

    R_optimizer.zero_grad()
    R_loss.backward()
    R_optimizer.step()

    return R_outputs, D_outputs


def main():
    args = set_parser()

    # Read translated files
    train_features = np.load("dump/train_features.npy")
    test_features = np.load("dump/test_features.npy")
    train_labels = np.load("dump/train_labels.npy")
    D_data = np.load("dump/disc_data.npy")
    if args.is_test_label:
        test_labels = np.load("dump/test_labels.npy")
    if args.is_val_label:
        val_labels = np.load("dump/val_labels.npy")

    hparam_suffix = "e{}_dr{}_wd_{}_tlr{}_dlr{}_lamda{}".format(\
        args.epoch, args.dr_rate, args.wd, args.task_lr, args.disc_lr, args.lambda_)

    tensor_test_features = torch.tensor(test_features, requires_grad=True).to(device)
    tensor_train_features = torch.tensor(train_features, requires_grad=True).to(device)
    y_train = torch.tensor(train_labels, requires_grad=False).to(device)
    tensor_D_labels = torch.tensor(D_data, requires_grad=False).to(device)
    if args.is_test_label:
        y_test = torch.tensor(test_labels,  requires_grad=False).to(device)
    if args.is_val_label:
        y_val = torch.tensor(val_labels,  requires_grad=False).to(device)

    merge_features = np.concatenate([test_features, train_features], axis=0)

    # input of models
    X = torch.arange(train_features.shape[0] + test_features.shape[0])
    X_train = X[test_features.shape[0]:].to(device)
    X_test = X[:test_features.shape[0]].to(device)

    R = RoleModel(init_features=merge_features, dr_rate=args.dr_rate, class_num=len(y_train.unique())).to(device)
    D = Discriminator(args.dr_rate, emb_size=merge_features.shape[1]).to(device)

    # loss and optimizer
    D_criterion = nn.BCELoss()
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.disc_lr,  weight_decay=args.wd)

    R_criterion = nn.CrossEntropyLoss() 
    R_optimizer = torch.optim.Adam(R.parameters(), lr=args.task_lr, weight_decay=args.wd)

    for e in range(args.epoch):
        # discriminator optimization
        R_outputs, D_outputs = train(D, D_criterion, tensor_D_labels, D_optimizer, R, R_criterion, y_train, R_optimizer, args.lambda_, X_train)

        R.eval()
        test_outputs = R(X_test)
        train_accuracy = get_accuracy(R_outputs, y_train)
        train_micro_f1, train_macro_f1 = get_f1(R_outputs, y_train)
        disc_accuracy = accuracy(D_outputs, tensor_D_labels)
        if args.is_val_label:
            val_outputs = test_outputs[:val_labels.shape[0]]
            test_outputs = test_outputs[val_labels.shape[0]:]
            val_accuracy = get_accuracy(val_outputs, y_val)
            val_micro_f1, val_macro_f1 = get_f1(val_outputs, y_val)
        if args.is_test_label:
            test_accuracy = get_accuracy(test_outputs, y_test)
            test_micro_f1, test_macro_f1 = get_f1(test_outputs, y_test)

    _logger.info(hparam_suffix)
    _logger.info("train_accuracy: {}".format(train_accuracy))
    if args.is_test_label:
        _logger.info("test_accuracy: {}".format(test_accuracy))
    if args.is_val_label:
        _logger.info("val accuracy: {}".format(val_accuracy))

    if args.param_dir is not None:
        after_test_features = R.embed.weight.detach()[:test_features.shape[0], :].cpu().numpy()
        after_train_features = R.embed.weight.detach()[test_features.shape[0]:, :].cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()
        save_params(args.param_dir, hparam_suffix, after_train_features, after_test_features, test_outputs)


if __name__ == '__main__':
    main()
