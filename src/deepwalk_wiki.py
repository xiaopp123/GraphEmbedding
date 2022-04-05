# -*- coding: utf-8 -*-


import os
import networkx as nx

from config import *
from src.model.deep_walk import DeepWalk
from src.model.classifier import Classifier
from sklearn.linear_model import LogisticRegression
from src.utils import read_node_label


def evaluate_embedding(embedding_dict):
    all_x, all_y = read_node_label(os.path.join(DATA_PATH, 'wiki/wiki_labels.txt'))
    cls = Classifier(embedding=embedding_dict, cls=LogisticRegression())
    cls.split_train_evaluate(all_x, all_y, train_percent=0.8, seed=0)


def build_network():
    wiki_path = os.path.join(DATA_PATH, 'wiki/Wiki_edgelist.txt')
    # wiki_edgelist.txt: from_vertix, to_vertix
    # 构建有向无环图
    graph = nx.read_edgelist(wiki_path, create_using=nx.DiGraph(),
                             data=[('weight', int)])
    # print(G)
    # print(G.edges(data=True))
    return graph


def main():
    # 1. 构图
    graph_network = build_network()
    # 2. 创建deepwalk实例，这里会生成节点序列
    deep_walk = DeepWalk(graph=graph_network, walk_length=10, num_walks=2)
    # 3. 训练word2vec
    deep_walk.train()
    embedding_dict = deep_walk.get_embedding()
    # 4. 使用节点embedding训练分类模型
    evaluate_embedding(embedding_dict)


if __name__ == '__main__':
    main()
