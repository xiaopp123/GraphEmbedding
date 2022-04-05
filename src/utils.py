# -*- coding: utf-8 -*-


import os
import networkx as nx
from config import *


def build_wiki_network():
    wiki_path = os.path.join(DATA_PATH, 'wiki/Wiki_edgelist.txt')
    # wiki_edgelist.txt: from_vertix, to_vertix
    # 构建有向无环图
    graph = nx.read_edgelist(wiki_path, create_using=nx.DiGraph(),
                             data=[('weight', int)])
    # print(G)
    # print(G.edges(data=True))
    return graph


def read_node_label(file_name):
    node_list = []
    label_list = []
    with open(file_name, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            node_list.append(line[0])
            label_list.append(line[1:])
    return node_list, label_list


def main():
    file_name = '../data/wiki/wiki_labels.txt'
    read_node_label(file_name)


if __name__ == '__main__':
    main()
