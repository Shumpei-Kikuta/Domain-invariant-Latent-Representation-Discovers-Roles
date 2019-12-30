
import torch
import torch.nn as nn
import numpy as np
import os
from model import RoleModel, Discriminator
from util import *
import argparse
import logging


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_target_label", action="store_true",
                        help="True if target labels are available for calculating target accuracy")

    parser.add_argument("--is_val_label", action="store_true",
                        help="True if validation labels are available for calculating validation accuracy")

    parser.add_argument("--param_dir", default=None,
                        help="preserve features and predicted labels")

    parser.add_argument("--epoch", default=3000, type=int,
                        help="How many times models iterate")

    parser.add_argument("--dr_rate", default=0.5, type=float,
                        help="Drop out rate should be set from 0 to 1")

    parser.add_argument("--lambda_", default=10, type=float,
                        help="Balance coefficient")

    parser.add_argument("--wd", default=1e-4, type=float,
                        help="weight decay rate")

    parser.add_argument("--r_lr", default=1e-4, type=float,
                        help="learning rate for Role-Model")

    parser.add_argument("--d_lr", default=1e-4, type=float,
                        help="learning rate for Discriminator")

    parser.add_argument("--suffix", default="adv", type=str, 
                        help="Suffix for this implementation, used for saving params, logging")

    args = parser.parse_args()
    return args


def save_params(param_dir, suffix, after_source_X, after_target_X, target_outputs):
    if not os.path.exists(param_dir):
        os.mkdir(param_dir)
    np.save(param_dir+"/source_feature_{}.npy".format(suffix), after_source_X)
    np.save(param_dir+"/target_feature_{}.npy".format(suffix), after_target_X)
    np.save(param_dir+"/target_outputs_{}.npy".format(suffix), target_outputs)


def train(D, D_criterion, D_labels, D_optimizer, R, R_criterion, y_source, R_optimizer, lambda_, X_source):
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
    R_outputs = R(X_source)

    move_weight = R.embed.weight
    D_outputs = D(move_weight)

    R_loss = R_criterion(R_outputs, y_source.long()) \
                - lambda_ * D_criterion(D_outputs, D_labels)

    R_optimizer.zero_grad()
    R_loss.backward()
    R_optimizer.step()

    return R_outputs, D_outputs


def main():
    args = set_parser()

    # Read latent representations
    source_X = np.load("dump/source_features.npy")
    target_X = np.load("dump/target_features.npy")
    merge_X = np.concatenate([target_X, source_X], axis=0)

    # Read labels
    source_labels = np.load("dump/source_labels.npy")
    y_source = torch.tensor(source_labels, requires_grad=False).to(device)
    if args.is_target_label:
        target_labels = np.load("dump/target_labels.npy")
        y_target = torch.tensor(target_labels, requires_grad=False).to(device)
    if args.is_val_label:
        val_labels = np.load("dump/val_labels.npy")
        y_val = torch.tensor(val_labels, requires_grad=False).to(device)
        val_size = val_labels.shape[0]

    # Read source or target label
    D_labels = np.load("dump/disc_data.npy")
    D_labels = torch.tensor(D_labels, requires_grad=False).to(device)

    target_size = target_X.shape[0]

    # input of models
    X = torch.arange(merge_X.shape[0])
    X_source = X[target_size:].to(device)
    X_target = X[:target_size].to(device)
    
    # model definition
    R = RoleModel(init_features=merge_X, dr_rate=args.dr_rate, class_num=len(y_source.unique())).to(device)
    D = Discriminator(args.dr_rate, emb_size=merge_X.shape[1]).to(device)

    # loss and optimizer
    D_criterion = nn.BCELoss()
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr,  weight_decay=args.wd)

    R_criterion = nn.CrossEntropyLoss()
    R_optimizer = torch.optim.Adam(R.parameters(), lr=args.r_lr, weight_decay=args.wd)

    for e in range(args.epoch):
        # train epoch
        R_outputs, D_outputs = train(D, D_criterion, D_labels, D_optimizer, R, R_criterion, y_source, R_optimizer, args.lambda_, X_source)

        # validation epoch
        R.eval()
        target_outputs = R(X_target)
        source_accuracy = get_accuracy(R_outputs, y_source)
        disc_accuracy = accuracy(D_outputs, D_labels)

        if args.is_val_label:
            val_outputs = target_outputs[:val_size]
            target_outputs = target_outputs[val_size:]
            val_accuracy = get_accuracy(val_outputs, y_val)
        if args.is_target_label:
            target_accuracy = get_accuracy(target_outputs, y_target)

    # log
    _logger.info(args.suffix)
    _logger.info("source_accuracy: {}".format(source_accuracy))
    _logger.info("disc accuracy: {}".format(disc_accuracy))
    if args.is_target_label:
        _logger.info("target_accuracy: {}".format(target_accuracy))
    if args.is_val_label:
        _logger.info("val accuracy: {}".format(val_accuracy))

    # save params
    if args.param_dir is not None:
        after_target_X = R.embed.weight.detach()[:target_size, :].cpu().numpy()
        after_source_X = R.embed.weight.detach()[target_size:, :].cpu().numpy()
        target_outputs = target_outputs.detach().cpu().numpy()
        save_params(args.param_dir, args.suffix, after_source_X, after_target_X, target_outputs)


if __name__ == '__main__':
    main()
