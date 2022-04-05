# -*- coding: utf-8 -*-


import os
import networkx as nx
from src.model.node2vec import Node2Vec
from src.model.classifier import Classifier
from sklearn.linear_model import LogisticRegression
from src.utils import read_node_label, build_wiki_network

from config import *


def evaluate_embedding(embedding_dict):
    # 读取每个节点及标签 <node_id, label>
    all_x, all_y = read_node_label(os.path.join(DATA_PATH, 'wiki/wiki_labels.txt'))
    # 逻辑回归多分类器
    cls = Classifier(embedding=embedding_dict, cls=LogisticRegression())
    # 划分训练集，在验证集测试效果
    cls.split_train_evaluate(all_x, all_y, train_percent=0.8, seed=0)


def main():
    # 1. 构图
    graph_network = build_wiki_network()
    # 2. 创建node2vec实例，这里会生成节点序列
    node2vec = Node2Vec(graph_network, walk_length=10, num_walks=80,
                        p=0.25, q=4, use_rejection_sampling=0)
    # 3. 训练word2vec
    node2vec.train()
    embedding_dict = node2vec.get_embedding()
    # 4. 使用节点embedding训练分类模型
    evaluate_embedding(embedding_dict)


if __name__ == '__main__':
    main()
