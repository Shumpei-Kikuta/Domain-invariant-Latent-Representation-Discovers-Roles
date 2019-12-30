
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
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
)
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_test_label", action="store_true", help="True if test labels are available for calculating test accuracy")
    parser.add_argument("--is_val_label", action="store_true", help="True if validation labels are available for calculating validation accuracy")
    parser.add_argument("--param_dir", default=None, help="preserve features and predicted labels")
    parser.add_argument("--epoch", default=3000, type=int, help="How many times models iterate")
    parser.add_argument("--dr_rate", default=0.5, type=float, help="Drop out rate should be set from 0 to 1")
    parser.add_argument("--lambda_", default=10, type=float, help="Balance coefficient")
    parser.add_argument("--wd", default=1e-4, type=float, help="weight decay rate")
    parser.add_argument("--task_lr", default=1e-4, type=float, help="learning rate for Role-Model")
    parser.add_argument("--disc_lr", default=1e-4, type=float, help="learning rate for Discriminator")
    args = parser.parse_args()
    return args


def save_params():
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    np.save(args.param_dir+"/train_feature_{}.npy".format(hparam_suffix), after_train_features)
    np.save(args.param_dir+"/test_feature_{}.npy".format(hparam_suffix), after_test_features)
    np.save(args.param_dir+"/test_outputs_{}.npy".format(hparam_suffix), test_outputs)


args = set_parser()

# Read translated files
train_features = np.load("dump/train_features.npy")
test_features = np.load("dump/test_features.npy")
train_labels = np.load("dump/train_labels.npy")
disc_data = np.load("dump/disc_data.npy")
if args.is_test_label:
    test_labels = np.load("dump/test_labels.npy")
if args.is_val_label:
    val_labels = np.load("dump/val_labels.npy")

hparam_suffix = "e{}_dr{}_wd_{}_tlr{}_dlr{}_lamda{}".format(args.epoch, args.dr_rate, args.wd, args.task_lr, args.disc_lr, args.lambda_)

tensor_test_features = torch.tensor(test_features, requires_grad=True).to(device)
tensor_train_features = torch.tensor(train_features, requires_grad=True).to(device)
y_train = torch.tensor(train_labels, requires_grad=False).to(device)
tensor_disc_labels = torch.tensor(disc_data, requires_grad=False).to(device)
if args.is_test_label:
    y_test = torch.tensor(test_labels,  requires_grad=False).to(device)
if args.is_val_label:
    y_val = torch.tensor(val_labels,  requires_grad=False).to(device)

merge_features = np.concatenate([test_features, train_features], axis=0)

# # モデルの入力
X = torch.arange(train_features.shape[0] + test_features.shape[0])
X_train = X[test_features.shape[0]:].to(device)
X_test = X[:test_features.shape[0]].to(device)

role_model = RoleModel(init_features=merge_features, dr_rate=args.dr_rate, class_label_num=len(y_train.unique())).to(device)
discriminator = Discriminator(args.dr_rate, EMB_SIZE=merge_features.shape[1]).to(device)

# loss and optimizer
disc_criterion = nn.BCELoss()
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.disc_lr,  weight_decay=args.wd)

task_criterion = nn.CrossEntropyLoss() 
task_optimizer = torch.optim.Adam(role_model.parameters(), lr=args.task_lr, weight_decay=args.wd)

for e in range(args.epoch):
    # discriminator optimization
    discriminator.train()
    
    detached_weight = role_model.embed.weight.detach()
    disc_outputs = discriminator(detached_weight)
    
    disc_loss = args.lambda_ * disc_criterion(disc_outputs, tensor_disc_labels)
    
    disc_optimizer.zero_grad()
    disc_loss.backward(retain_graph=True)
    disc_optimizer.step()
    
    # task optimization
    role_model.train()
    task_outputs = role_model(X_train)

    move_weight = role_model.embed.weight
    disc_outputs = discriminator(move_weight)

    task_loss = task_criterion(task_outputs, y_train.long()) -  args.lambda_ * disc_criterion(disc_outputs, tensor_disc_labels)
    
    task_optimizer.zero_grad()
    task_loss.backward()
    task_optimizer.step()

    role_model.eval()
    test_outputs = role_model(X_test)
    train_accuracy = get_accuracy(task_outputs, y_train)
    train_micro_f1, train_macro_f1 = get_f1(task_outputs, y_train)
    disc_accuracy = accuracy(disc_outputs, tensor_disc_labels)
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
    after_test_features = role_model.embed.weight.detach()[:test_features.shape[0], :].cpu().numpy()
    after_train_features = role_model.embed.weight.detach()[test_features.shape[0]:, :].cpu().numpy()
    test_outputs = test_outputs.detach().cpu().numpy()
    save_params(args.param_dir, hparam_suffix, after_train_features, after_test_features, test_outputs)
