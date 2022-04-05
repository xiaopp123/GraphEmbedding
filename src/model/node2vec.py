# -*- coding: utf-8 -*-


from gensim.models import Word2Vec
from src.model.random_walk import RandomWalk


class Node2Vec(object):
    def __init__(self, graph, walk_length, num_walks, p, q,
                 use_rejection_sampling):
        self.graph = graph
        # 序列长度
        self.walk_length = walk_length
        # 每个节点开始的次数
        self.num_walks = num_walks
        self.p = p
        self.q = q
        # 是否拒绝采样
        self.use_rejection_sampling = use_rejection_sampling
        # 游走实例
        self.walker = RandomWalk(graph=self.graph, p=self.p, q=self.q,
                                 use_rejection_sampling=0)
        # 构建采样表
        self.walker.preprocess_transition_prob()
        # 根据游走算法生成节点序列
        self.sentence_list = self.walker.naive_walks(
            num_walks=self.num_walks, walk_length=self.walk_length)
        # word2vec
        self.w2v_model = None
        # 节点embedding表
        self._embedding_dict = dict()

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        """
        根据生成的节点训练，使用word2vec训练embedding
        :param embed_size:
        :param window_size:
        :param workers:
        :param iter:
        :param kwargs:
        :return:
        """
        kwargs["sentences"] = self.sentence_list
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        # kwargs["iter"] = iter
        self.w2v_model = Word2Vec(**kwargs)

    def get_embedding(self):
        if not self.w2v_model:
            return {}
        for node in self.graph.nodes():
            self._embedding_dict[node] = self.w2v_model.wv[node]
        return self._embedding_dict