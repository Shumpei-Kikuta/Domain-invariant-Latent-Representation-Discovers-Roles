import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch


def exec_tsne(train_X, test_X):
    train_shape = train_X.shape
    X = np.concatenate([train_X, test_X], axis=0)

    assert(X.shape[0] == train_X.shape[0] + test_X.shape[0])

    tsne = TSNE(n_components=2)
    tsne_X = tsne.fit_transform(X)
    tsne_train_X, tsne_test_X = tsne_X[:train_shape[0], :], tsne_X[train_shape[0]:, :]

    assert(tsne_train_X.shape[0] == train_X.shape[0])

    return tsne_train_X, tsne_test_X


def exec_pca(train_X, test_X):
    pca = PCA(n_components=2)
    pca_train_X = pca.fit_transform(train_X)
    pca_test_X = pca.transform(test_X)

    return pca_train_X, pca_test_X


def plot_two_scatter(train_X, test_X):
    plt.figure(figsize=(10, 10))
    plt.scatter(x=train_X[:, 0], y=train_X[:, 1], c="blue")
    plt.scatter(x=test_X[:, 0],  y=test_X[:, 1], c="red")


def get_f1(output: torch.tensor, labels: torch.tensor) -> float:
    """
    INPUT:
        output: (N, CLASS_LABEL_NUM)  # one-hot expression
        labels (N, )  # raw expression
    OUTPUT:
        micro_f1: float
        macro_f1: float
    """
    from sklearn.metrics import f1_score
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    output = np.argmax(output, axis=1)
    assert(output.shape[0] == labels.shape[0])
    micro_f1 = f1_score(labels, output, average="micro")
    macro_f1 = f1_score(labels, output, average="macro")
    return micro_f1, macro_f1


def get_accuracy(output: torch.tensor, labels: torch.tensor) -> float:
    """
    INPUT:
        output: (N, CLASS_LABEL_NUM)  # one-hot expression
        labels (N, )  # raw expression
    OUTPUT:
        accuracy: float
    """
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    output = np.argmax(output, axis=1)
    assert(output.shape[0] == labels.shape[0])
    accuracy = np.sum(output == labels) / labels.shape[0]
    return accuracy


def accuracy(output: torch.tensor, labels: torch.tensor):
    """
    INPUT:
        output: (N, 1), labels: (N, 1)
    OUTPUT:
        accuracy: float
    """
    output = output.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    # TODO: Flattenいる？
    output = output.flatten()

    assert(output.shape[0] == labels.shape[0])
    output = output > 0.5
    accuracy = np.sum(output == labels) / labels.shape[0]

    return accuracy


def split_cross_val(X, cv_num=5):
    data_num = X.shape[0]
    devided_id = np.arange(0, data_num)
    np.random.shuffle(devided_id)
    cv_size = data_num // cv_num
    train_val_dicts = {"train": [], "test": []}  # {train: [[] * 5,] test: [[] * 5]}
    for i in range(cv_num):
        val_id = devided_id[i * cv_size: (i + 1) * cv_size] if i != cv_num - 1 else devided_id[i * cv_size:]
        train_id = [id_ for id_ in devided_id if id_ not in val_id]
        train_val_dicts["train"].append(train_id)
        train_val_dicts["test"].append(val_id)
    return train_val_dicts


class Dotdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
