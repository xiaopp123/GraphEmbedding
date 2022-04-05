# -*- coding: utf-8 -*-


from gensim.models import Word2Vec
from src.model.random_walk import RandomWalk


class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks):
        self.graph = graph
        # 序列长度
        self.walk_length = walk_length
        # 每个节点开始的次数
        self.num_walks = num_walks
        # 随机游走实例
        self.walker = RandomWalk(graph=self.graph, p=1, q=1,
                                 use_rejection_sampling=0)
        # 根据随机游走算法生成节点序列
        self.sentence_list = self.walker.naive_walks(
            num_walks=self.num_walks, walk_length=self.walk_length)
        # word2vec
        self.w2v_model = None
        # 节点embedding表
        self._embedding_dict = dict()

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

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

