# -*- coding: utf-8 -*-

import io
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
from pickle_tools import load_pickle, dump_pickle
import numpy as np


USING_WORDS_NUM = 2000000
WORD_CLUSTERS_NUM = 1000
USING_ARTICLES_NUM = 100
ARTICLE_CLUSTERS_NUM = 10


def _cos_sim(v1, v2):
    """
    2つのベクトルのコサイン類似度を返す
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _centroid(vec_list):
    centroid_vec = []
    for i in range(len(vec_list[0])):
        centroid_vec.append(sum(list(map(lambda x: x[i], vec_list))) / len(vec_list))

    return np.array(centroid_vec)


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


def make_cluster_dict(data, clusters, data_num):
    """
    {"単語1": 単語ベクトル1, "単語2", 単語ベクトル2・・・}のようなdictとK-Meansのクラスタ数を受け取って,
    {"単語1": クラスタラベル1, "単語2": クラスタラベル2・・・}のようなdictを返す.
    """
    x_tuple_list = []
    x_list = []
    for i, d in enumerate(data):
        if i == data_num:
            break
        x_tuple_list.append((d, data[d]))
        x_list.append((data[d]))
    
    print("x_list[0] = ", x_list[0])

    k_means = MiniBatchKMeans(n_clusters=clusters, verbose=1, n_init=1)
    pred = k_means.fit_predict(x_list)

    cluster_res = {}

    for x_tuple, label in zip(x_tuple_list, pred):
        cluster_res[x_tuple[0]] = label

    return cluster_res


def document_to_vec(document, cluster_dict):
    """
    半角スペースで区切られた文書と, 単語クラスタdictから, 文書を表すベクトル(list)を返す. 
    :param document: 
    :param cluster_dict: 
    :return: 
    """
    word_list = document.split(" ")
    cluster_cnt_dict = defaultdict(int)
    all_cnt = 0
    returning_vec = []
    for w in word_list:
        if w in cluster_dict:
            cluster_cnt_dict[cluster_dict[w]] += 1
            all_cnt += 1
    if all_cnt != 0:
        for i in range(WORD_CLUSTERS_NUM):
            returning_vec.append(cluster_cnt_dict[i] / all_cnt)
    else:
        returning_vec = [0 for i in range(WORD_CLUSTERS_NUM)]

    print(returning_vec)
    return returning_vec
    

def online_cluster_docs(document_vec_tuple_list):
    """
    (見出し, 半角スペース区切り本文, 文書分散表現)のtupleのリストを受け取って,
    {"included_docs": [tuple, tuple, ...], "centroid": 重心ベクトル(list)}のdictのlistを返す.
    :param document_vec_tuple_list: 
    :return: 
    """

    CLUSTER_THRESHOLD = 0.9

    clusters_dict_list = []

    for i, document_vec_tuple in enumerate(document_vec_tuple_list, start=1):
        print("online clustering... {} / {}".format(i, len(document_vec_tuple_list)))
        headline = document_vec_tuple[0]
        document = document_vec_tuple[1]
        document_vec = np.array(document_vec_tuple[2])

        cos_sim_list = []
        print("clusters_num", len(clusters_dict_list))
        for clusters_dict in clusters_dict_list:
            cos_sim_list.append(_cos_sim(clusters_dict['centroid'], list(document_vec)))

        # print(cos_sim_list)

        if len(cos_sim_list) > 0:
            closest_cluster_num = np.array(cos_sim_list).argmax()
            closest_cluster_sim = np.array(cos_sim_list).max()

            print("sim = ", closest_cluster_sim)

            if closest_cluster_sim > CLUSTER_THRESHOLD:
                print("はいってる", closest_cluster_num)
                clusters_dict_list[closest_cluster_num]['included_docs'].append(document_vec_tuple)
                document_vec_list = list(map(lambda x: x[2], clusters_dict_list[closest_cluster_num]['included_docs']))
                clusters_dict_list[closest_cluster_num]['centroid'] = _centroid(document_vec_list)
            else:
                clusters_dict_list.append({'included_docs': [document_vec_tuple],
                                           'centroid'     : document_vec})
        else:
            clusters_dict_list.append({'included_docs': [document_vec_tuple],
                                       'centroid': document_vec})

    return clusters_dict_list


def mainichi_corpus_data_to_documents(filename):
    """
    毎日新聞コーパスデータのpickleファイル名を受け取って,
    (記事見出し, 単語ごとに半角スペースで区切られた記事全文)のtupleのリストを返す.
    """
    article_list = load_pickle(filename)
    headline_document_tuple_list = []
    for i, article in enumerate(article_list, start=1):
        print(i)
        word_list = []
        article_headline = article["t1"][0]
        for spd in article["_sentence_parse_dict_list"]:
            for p in spd["parse"]:
                if p["surface"] != "\u3000":
                    word_list.append(p["surface"])
        document = " ".join(word_list)
        headline_document_tuple_list.append((article_headline, document))
    
    return headline_document_tuple_list


def main():
    # data = load_vectors("cc.ja.300.vec")
    # word_cluster_dict = make_cluster_dict(data, WORD_CLUSTERS_NUM, USING_WORDS_NUM)
    # dump_pickle(word_cluster_dict, "word_cluster_dict_mini.pickle")

    word_cluster_dict = load_pickle("word_cluster_dict_mini.pickle")

    # for i in range(WORD_CLUSTERS_NUM):
    #     print("label = {}".format(i))
    #     for w in word_cluster_dict:
    #         if word_cluster_dict[w] == i:
    #             print(w, end=", ")
    #     print("-"*100)

    headline_document_tuple_list = mainichi_corpus_data_to_documents("/home/ytaniguchi/kenkyu/news_systematize_2/corpus_data/pickles/mai2017_word_parse_added_part2.pickle")
    document_vec_list = []
    
    for headline_document_tuple in headline_document_tuple_list[:USING_ARTICLES_NUM]:
        document_vec = document_to_vec(headline_document_tuple[1], word_cluster_dict)
        document_vec_list.append((headline_document_tuple[0], headline_document_tuple[1], document_vec))
    dump_pickle(document_vec_list, "document_vec_list_mini.pickle")

    # print("document_vec_list", document_vec_list[0])

    document_vec_dict = {}

    for document_vec_tuple in document_vec_list:
        document_vec_dict[document_vec_tuple[0]] = document_vec_tuple[2]

    print(len(document_vec_list))
    print(len(document_vec_dict))

    doc_cluster_dict_list = online_cluster_docs(document_vec_list)

    for i, doc_cluster_dict in enumerate(doc_cluster_dict_list, start=1):
        print("label = {}".format(i))
        for included_doc in doc_cluster_dict["included_docs"]:
            print(included_doc[0])
        print("-"*100)

    # doc_cluster_dict = make_cluster_dict(document_vec_dict, ARTICLE_CLUSTERS_NUM, USING_ARTICLES_NUM)

    # for i in range(ARTICLE_CLUSTERS_NUM):
    #     print("label = {}".format(i))
    #     for d in doc_cluster_dict:
    #         if doc_cluster_dict[d] == i:
    #             print(d)
    #     print("-"*100)

    # cluster_dict = load_pickle("word_cluster_dict.pickle")
    # vec = document_to_vec("私 は 元から 強い 。", cluster_dict)
    # print(vec)
    # vec = document_to_vec("私 は 古い 人間 だ 。", cluster_dict)
    # print(vec)


if __name__ == "__main__":
    main()
