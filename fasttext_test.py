# -*- coding: utf-8 -*-

import io
from sklearn.cluster import KMeans
from collections import defaultdict
from pickle_tools import load_pickle, dump_pickle


USING_WORDS_NUM = 2000000
CLUSTERS_NUM = 1000


def load_vectors(fname):
    """
    FastTextの学習済みモデルをファイル名を受け取って,
    {"単語1": 単語ベクトル1, "単語2", 単語ベクトル2・・・} のようなdictを返す.
    """
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = list(map(int, fin.readline().split()))
    data = {}
    for i, line in enumerate(fin, start=1):
        if i == USING_WORDS_NUM + 1:
            break
        print("{} / {}".format(i, n))
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def make_word_cluster_dict(data, clusters):
    """
    {"単語1": 単語ベクトル1, "単語2", 単語ベクトル2・・・}のようなdictとK-Meansのクラスタ数を受け取って,
    {"単語1": クラスタラベル1, "単語2": クラスタラベル2・・・}のようなdictを返す.
    """
    x_tuple_list = []
    x_list = []
    for i, d in enumerate(data):
        if i == USING_WORDS_NUM:
            break
        x_tuple_list.append((d, data[d]))
        x_list.append((data[d]))
    
    k_means = KMeans(n_clusters=CLUSTERS_NUM)
    pred = k_means.fit_predict(x_list)

    cluster_res = {}

    for x_tuple, label in zip(x_tuple_list, pred):
        cluster_res[x_tuple[0]] = label

    return cluster_res


def document_to_vec(document, cluster_dict):
    word_list = document.split(" ")
    cluster_cnt_dict = defaultdict(int)
    all_cnt = 0
    returning_vec = []
    for w in word_list:
        if w in cluster_dict:
            cluster_cnt_dict[cluster_dict[w]] += 1
            all_cnt += 1
            
    print(cluster_cnt_dict)
    
    for i in range(CLUSTERS_NUM):
        returning_vec.append(cluster_cnt_dict[i] / all_cnt)
        
    return returning_vec


def main():
    data = load_vectors("cc.ja.300.vec")
    word_cluster_dict = make_word_cluster_dict(data, CLUSTERS_NUM)
    dump_pickle(word_cluster_dict, "word_cluster_dict.pickle")
    
# cluster_dict = load_pickle("word_cluster_dict.pickle")
    # vec = document_to_vec("私 は 元から 強い 。", cluster_dict)
    # print(vec)
    # vec = document_to_vec("私 は 古い 人間 だ 。", cluster_dict)
    # print(vec)


if __name__ == "__main__":
    main()
