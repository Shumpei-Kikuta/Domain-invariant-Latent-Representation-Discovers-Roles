"""
embファイル、componentファイル、labelファイルから、modelが欲しい形に翻訳する
INPUT: embファイル、componentファイル、labelファイル(2つ)
"""

import argparse
import numpy as np
import pandas as pd
import os
import logging


_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s"
)


def specify_which(source_labels, val_labels, target_labels, disc_data):
    """disc_dataの中身を解析し、disc_dataのどっちがsourceでどっちがテストか"""
    # TODO: refactoring
    if source_labels.shape[0] == target_labels.shape[0]:
        _logger.warning("target and source size is the same. Check to see source and target is separated as you want.")
    label0_num = disc_data[disc_data.component_label == 0].shape[0]
    label1_num = disc_data[disc_data.component_label == 1].shape[0]
    label2_num = disc_data[disc_data.component_label == 2].shape[0]
    label_nums = [0, 1, 2]

    if label0_num == source_labels.shape[0]:
        source_label = 0
    elif label1_num == source_labels.shape[0]:
        source_label = 1
    elif label2_num == source_labels.shape[0]:
        source_label = 2
    else:
        raise ValueError("source label cannot be specified")
    label_nums.remove(source_label)

    if label0_num == val_labels.shape[0]:
        val_label = 0
    elif label1_num == val_labels.shape[0]:
        val_label = 1
    elif label2_num == val_labels.shape[0]:
        val_label = 2
    else:
        raise ValueError("Val label cannot be specified")
    label_nums.remove(val_label)

    target_label = label_nums[0]
    return source_label, val_label, target_label


def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", default="graph/labels-usa-airports.txt",
                        help="source labels file")
    parser.add_argument("--target", default=None,
                        help="target labels file")
    parser.add_argument("--validation", default=None,
                        help="Validation labels file")
    parser.add_argument("emb", default="emb/usa_europe.emb",
                        help="Emb file")
    parser.add_argument("component", default="graph/usa_europe.component",
                        help="Emb file")
    args = parser.parse_args()
    return args


def prepare_discriminator(disc_data: pd.DataFrame, source_label: int, val_label: int, target_label: int):
    source_disc = disc_data[disc_data.component_label == source_label].sort_values(by="node")
    val_disc = disc_data[disc_data.component_label == val_label].sort_values(by="node")
    target_disc = disc_data[disc_data.component_label == target_label].sort_values(by="node")

    source_disc.component_label = 0
    val_disc.component_label = 1
    target_disc.component_label = 1

    disc_data = pd.concat([target_disc, val_disc, source_disc], axis=0)   # [1, 1, ,1, ,1 , ....., 0, 0, 0] euなら1, usaなら0
    disc_data = disc_data.component_label.values.astype(np.float32)
    return disc_data


def prepare_features(features, source_label, val_label, target_label):
    source_features = features[features.component_label == source_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    val_features = features[features.component_label == val_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    target_features = features[features.component_label == target_label].sort_values(by=0).drop(["component_label", "node", 0], axis=1).values.astype(np.float32)
    # import pdb; pdb.set_trace()
    if val_features.shape[0] != 0:
        target_features = np.concatenate([val_features, target_features], axis=0)
    return source_features, target_features


def prepare_labels(source_labels, val_labels, target_labels):
    source_labels = source_labels.sort_values(by="node").label.values.astype(np.float32)
    val_labels = val_labels.sort_values(by="node").label.values.astype(np.float32)
    target_labels = target_labels.sort_values(by="node").label.values.astype(np.float32)
    return source_labels, val_labels, target_labels


def exec_translate(args):
    # 必要なファイルの読み込み
    features = pd.read_csv(args.emb, sep=" ", skiprows=1, header=None)
    disc_data = pd.read_csv(args.component, sep=" ")  # 0->usa, 1->eu
    source_labels = pd.read_csv(args.source, sep=" ")
    if args.target is not None:
        target_labels = pd.read_csv(args.target, sep=" ")
    else:
        target_labels = pd.DataFrame(columns=["label", "node"])
    
    if args.validation is not None:
        val_labels = pd.read_csv(args.validation, sep=" ")
    else:
        val_labels = pd.DataFrame(columns=["label", "node"])

    _logger.info("source_size: {}, val_size: {}, target_size: {}".format(
                source_labels.shape[0], val_labels.shape[0], target_labels.shape[0]))

    # sourceとtargetのラベルの設定
    source_label, val_label, target_label = specify_which(source_labels, val_labels, target_labels, disc_data)
    features = pd.merge(left=features, right=disc_data, left_on=0, right_on="node", how="inner")

    # それぞれinputの形へ
    source_features, target_features = prepare_features(features, source_label, val_label, target_label)
    disc_data = prepare_discriminator(disc_data, source_label, val_label, target_label)
    source_labels, val_labels, target_labels = prepare_labels(source_labels, val_labels, target_labels)

    # npyファイルで吐き出す
    if os.path.exists("dump"):
        import shutil
        shutil.rmtree("dump")
    os.makedirs("dump")

    np.save("dump/source_features.npy", source_features)
    np.save("dump/target_features.npy", target_features)
    np.save("dump/source_labels.npy", source_labels)
    np.save("dump/disc_data.npy", disc_data)
    if args.target is not None:
        np.save("dump/target_labels.npy", target_labels)
    if args.validation is not None:
        np.save("dump/val_labels.npy", val_labels)
    _logger.info("successfully saved numpy file!")


if __name__ == '__main__':
    args = set_parser()
    exec_translate(args)
