# -*- coding: utf-8 -*-


import random
from src.alias import create_alias_table, alias_sample


class RandomWalk(object):
    def __init__(self, graph, p=1, q=1, use_rejection_sampling=0):
        self.graph = graph
        # p和q参数是node2v
        self.p = p
        self.q = q
        # 拒绝采样暂未实现
        self.use_rejection_sampling = use_rejection_sampling
        # alias采样
        # alias_nodes，根据当前节点选择邻接点
        self.alias_nodes = None
        # alias_edges，根据当前节点和上一步节点选择当前节点邻接点
        self.alias_edges = None

    def deepwalk_walk(self, walk_length, start_node):
        """
        :param walk_length: 序列长度
        :param start_node:  开始节点
        :return:
        """
        walk_path = [start_node]
        while len(walk_path) < walk_length:
            cur_node = walk_path[-1]
            # 当前节点的邻接点
            neighbor_node_list = list(self.graph.neighbors(cur_node))
            if not neighbor_node_list:
                break
            # 随机选择一个邻接点
            selected_neighbor_node = random.choice(neighbor_node_list)
            walk_path.append(selected_neighbor_node)
        return walk_path

    def node2vec_walk(self, walk_length, start_node):
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        graph = self.graph
        walk_path = [start_node]

        while len(walk_path) < walk_length:
            cur_node = walk_path[-1]
            # 当前节点邻接点列表
            neighbor_node_list = list(graph.neighbors(cur_node))
            if not neighbor_node_list:
                break
            if len(walk_path) == 1:
                # 当前节点是起点时，就不用考虑上一个节点，直接重当前节点的邻接点中采样
                sample_node_id = alias_sample(
                    alias_nodes[cur_node][0], alias_nodes[cur_node][1])
            else:
                # 上一个节点t
                t = walk_path[-2]
                # 根据上一个节点t和当前节点v进行采样
                edge = (t, cur_node)
                sample_node_id = alias_sample(
                    alias_edges[edge][0], alias_edges[edge][1])
            walk_path.append(neighbor_node_list[sample_node_id])
        return walk_path

    def naive_walks(self, num_walks, walk_length):
        """ 朴素随机游走算法
        :param num_walks:
        :param walk_length:
        :return:
        """
        sentence_list = []
        node_list = list(self.graph.nodes())
        for i in range(num_walks):
            random.shuffle(node_list)
            # 每个节点作为开始节点进行遍历
            for node in node_list:
                if self.p == 1 and self.q == 1:
                    # deepwalk
                    sentence_list.append(self.deepwalk_walk(walk_length, node))
                elif self.use_rejection_sampling:
                    pass
                else:
                    # node2vec
                    sentence_list.append(self.node2vec_walk(walk_length, node))
        return sentence_list

    def get_alias_edge(self, t, v):
        p = self.p
        q = self.q
        graph = self.graph
        prob_list = []
        for x in graph.neighbors(v):
            # w_vx
            weight = graph[v][x].get('weight', 1.0)
            if x == t:
                #  d_tx == 0
                prob_list.append(weight/p)
            elif graph.has_edge(t, x):
                # d_tx == 1
                prob_list.append(weight)
            else:
                # d_tx == 2
                prob_list.append(weight/q)
        # 概率归一化
        norm_const = sum(prob_list)
        norm_prob_list = [float(prob) / norm_const for prob in prob_list]
        return create_alias_table(norm_prob_list)

    def preprocess_transition_prob(self):
        # alias_nodes存储当前节点到邻接点的转移概率
        alias_nodes = {}
        # alias_edges存储上一个节点是t,当前节点v到邻接点的转移概率
        alias_edges = {}
        for node in self.graph.nodes():
            # 当前节点到邻接点的权重
            prob_list = [self.graph[node][neigh_node].get('weight', 1.0)
                         for neigh_node in self.graph.neighbors(node)]
            # 归一化因子
            norm_const = sum(prob_list)
            # 归一化的概率
            norm_prob_list = [float(prob)/norm_const for prob in prob_list]
            # 按照alias算法构建采样表
            alias_nodes[node] = create_alias_table(norm_prob_list)

        for edge in self.graph.edges():
            # 上一个节点是edge[0],当前节点为edge[1]
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges





