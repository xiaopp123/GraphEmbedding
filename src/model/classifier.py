# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


class TopRanker(OneVsRestClassifier):
    def predict(self, x_test, top_k_list):
        prob_list = np.asarray(super(TopRanker, self).predict_proba(x_test))
        label_list = []
        for i, k in enumerate(top_k_list):
            prob = prob_list[i, :]
            label = self.classes_[prob.argsort()[-k:]].tolist()
            prob[:] = 0
            prob[label] = 1
            label_list.append(prob)
        return np.asarray(label_list)


class Classifier(object):
    def __init__(self, embedding, cls):
        self.embedding = embedding
        self.cls = TopRanker(cls)
        # self.cls = cls
        # multi-label编码
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def train(self, x_train, y_train, all_y):
        # 标签编码
        self.binarizer.fit(all_y)
        # 根据训练集中node查找其embedding
        x_train = [self.embedding[x] for x in x_train]
        y = self.binarizer.transform(y_train)
        # print(y_train)
        # 使用训练集训练模型
        self.cls.fit(X=x_train, y=y)

    def evaluate(self, x_test, y_test):
        # 每个测试样本label的个数，这里都是1
        top_k_list = [len(l) for l in y_test]
        y_pred = self.predict(x_test, top_k_list=top_k_list)
        print('y_pred:\n', y_pred)
        y_test = self.binarizer.transform(y_test)
        result = {}
        averages = ["micro", "macro", "samples", "weighted"]
        for ave in averages:
            result[ave] = f1_score(y_test, y_pred, average=ave)
        result['acc'] = accuracy_score(y_test, y_pred)
        print(result)
        pass

    def predict(self, x_test, top_k_list):
        x_test = np.asarray([self.embedding[x] for x in x_test])
        return self.cls.predict(x_test, top_k_list)

    def split_train_evaluate(self, X, Y, train_percent, seed=0):
        # 按8：2划分训练集和测试集
        train_size = int(len(X) * train_percent)
        shuffle_indices = np.random.permutation(np.arange(len(X)))
        x_train = [X[shuffle_indices[i]] for i in range(train_size)]
        y_train = [Y[shuffle_indices[i]] for i in range(train_size)]
        x_test = [X[shuffle_indices[i]] for i in range(train_size, len(X))]
        y_test = [Y[shuffle_indices[i]] for i in range(train_size, len(X))]

        # 训练
        self.train(x_train, y_train, Y)
        # 测试
        return self.evaluate(x_test, y_test)


def main():
    pass


if __name__ == '__main__':
    main()
