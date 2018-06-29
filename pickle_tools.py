#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pickle


def load_pickle(pickle_name):
    print("loading pickle...", pickle_name)
    article_list = pickle.load(open(pickle_name, mode="rb"))
    print("loaded.")
    return article_list


def dump_pickle(saving, pickle_name):
    print("dumping pickle...", pickle_name)
    with open(pickle_name, mode="wb") as f:
        pickle.dump(saving, f)
    print("dumped.")
    