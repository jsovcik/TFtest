"""Utilities for reading text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tagging as tg


import tensorflow as tf



def read_words(filename):
    "return a list of the words in filename"
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", " ").replace("'"," ").lower().split()

def make_sentences(s, tags):
    "divise a string list(typically a read_words output) into sentences"
    sent, sentences, sent_lengths = [], [], []
    tag, tg_sent = [], []
    cmp = 0
    for i in range(len(s)):
        sent.append(s[i])
        tg_sent.append(tags[i])
        cmp += 1
        if (cmp == 20):
            sentences.append(sent)
            tag.append(tg_sent)
            sent_lengths.append(cmp)
            sent = []
            tg_sent = []
            cmp = 0
    sentences.append(sent)
    tag.append(tg_sent)
    sent_lengths.append(cmp)
    return sentences, tag

def build_vocab(filename):
    "construit un lexique avec les mots dans le texte plus unk"
    data = _clean_str(read_words(filename))
    lex = {}
    cmp = 0
    for wrd in data:
        if not (wrd in lex):
            lex[wrd] = cmp
            cmp += 1
    lex ['unk'] = cmp
    return lex

def _clean_str(s):
    """enleve les mots sans lettres"""
    l = []
    for i in range(len(s)):
        if s[i].isalnum() and (not s[i].isdigit()):
            l.append(s[i])
    return l

def _replace_unk (txt, word_to_id):
    for word in txt:
        if word not in word_to_id:
            word = "unk"
    return txt


def lex_add (dic1, dic2):
    valuecumpt = len(dic1)+1
    comp = 0
    for word in dic2:
        if word not in dic1:
            dic1[word] = valuecumpt
            valuecumpt = valuecumpt+1
    return dic1

def write_metadata(lex, filepath):
    with open(filepath, "a") as f:
        f.write("mot\tindex\n")
        for w in lex:
            f.write(w.encode("utf-8") + "\t" + str(lex[w])+"\n")

def _file_to_word_ids(filename, word_to_id):
    """prend un fichier txt, un lexique et retourne la liste des id correspondant au texte"""
    data = _replace_unk(_clean_str(read_words(filename)), word_to_id)
    return [word_to_id[word] for word in data ]

def getTrData(filenames, NamedEntities, lex):
    tags = []
    txt = []
    for i in range(len(filenames)):
        tags += tg.preprocess_tr_data(filenames[i], NamedEntities[i])
        txt += _file_to_word_ids(filenames[i], lex)
    return make_sentences(txt, tags)

class Arg():
    def __init__(self, filenames, NamedEntities, embedding_size, lstm_size, batch_size, epoch):
        self.filenames = filenames
        self.NamedEntities = NamedEntities
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.lex = build_vocab(filenames[0])
        for fname in filenames[1:]:
            tmp = build_vocab(fname)
            self.lex = lex_add(self.lex, tmp)
        # rd.write_metadata(self.lex, "/home/jeremie/PycharmProjects/NER/data_tr/metadata.tsv")
        self.n_word = len(self.lex) + 1

if __name__ == '__main__':
    # args = Arg(["/home/jeremie/PycharmProjects/untitled/fichiertest.txt"],
    #         [[tg.NamedEntity(["bretagne", "telecom"], "C"), tg.NamedEntity(["alfresco", "heberge"], "P")]],
    #         25, 50, 15, 5)
    #print (getTrData(args.filenames, args.NamedEntities, args.lex))
    #print(read_words("/home/jeremie/PycharmProjects/untitled/fichiertest.txt"))
    pass