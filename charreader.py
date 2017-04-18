import codecs
import os
import collections
import pickle
import numpy as np


# class TextLoader():
#     def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.seq_length = seq_length
#         self.encoding = encoding
#
#         input_file = os.path.join(data_dir, "input.txt")
#         vocab_file = os.path.join(data_dir, "vocab.pkl")
#         tensor_file = os.path.join(data_dir, "data.npy")
#
#         if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
#             print("reading text file")
#             self.preprocess(input_file, vocab_file, tensor_file)
#         else:
#             print("loading preprocessed files")
#             self.load_preprocessed(vocab_file, tensor_file)
#         self.create_batches()
#         self.reset_batch_pointer()
#
#     def preprocess(self, input_file, vocab_file, tensor_file):
#         with codecs.open(input_file, "r", encoding=self.encoding) as f:
#             data = f.read()
#         counter = collections.Counter(data)
#         count_pairs = sorted(counter.items(), key=lambda x: -x[1])
#         self.chars, _ = zip(*count_pairs)
#         self.vocab_size = len(self.chars)
#         self.vocab = dict(zip(self.chars, range(len(self.chars))))
#         with open(vocab_file, 'wb') as f:
#             pickle.dump(self.chars, f)
#         self.tensor = np.array(list(map(self.vocab.get, data)))
#         np.save(tensor_file, self.tensor)
#
#     def load_preprocessed(self, vocab_file, tensor_file):
#         with open(vocab_file, 'rb') as f:
#             self.chars = pickle.load(f)
#         self.vocab_size = len(self.chars)
#         self.vocab = dict(zip(self.chars, range(len(self.chars))))
#         self.tensor = np.load(tensor_file)
#         self.num_batches = int(self.tensor.size / (self.batch_size *
#                                                    self.seq_length))
#
#     def create_batches(self):
#         self.num_batches = int(self.tensor.size / (self.batch_size *
#                                                    self.seq_length))
#
#         # When the data (tensor) is too small,
#         # let's give them a better error message
#         if self.num_batches == 0:
#             assert False, "Not enough data. Make seq_length and batch_size small."
#
#         self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
#         xdata = self.tensor
#         ydata = np.copy(self.tensor)
#         ydata[:-1] = xdata[1:]
#         ydata[-1] = xdata[0]
#         self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
#                                   self.num_batches, 1)
#         self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
#                                   self.num_batches, 1)
#
#     def next_batch(self):
#         x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
#         self.pointer += 1
#         return x, y
#
#     def reset_batch_pointer(self):
#         self.pointer = 0
def _read_char(filepath):
    with open(filepath) as f:
        data = f.read().decode("utf-8").lower()
    chars = []
    for i in data:
        chars.append(i)
    return chars

def _build_char_ind(filepath):
    data = _read_char(filepath)
    ind = {}
    comp = 0
    for c in data:
        if not c in ind:
            ind[c] = comp
            comp = comp +1
    return ind

def add_char_dict(dic1, dic2):
    valuecumpt = len(dic1) + 1
    for char in dic2:
        if char not in dic1:
            dic1[char] = valuecumpt
            valuecumpt = valuecumpt + 1
    return dic1

# fct test pour la longueur de padding

# class indic():
#     """sert just pour long_phr"""
#     def __init__(self, i, max):
#         self.indice = i
#         self.max = max
# def long_phrase(filepath):
#     """fct de test pour determiner la longueur de padding"""
#     with open(filepath) as f:
#         data = f.read().split(".")
#     i = indic("", 0)
#     for i.indice in data :
#         c = len(i.indice)
#         if i.max < c :
#             i.max = c
#     return i



