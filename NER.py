from __future__ import division

import tensorflow as tf
import numpy as np
import tagging as tg
import reader as rd
from tensorflow.contrib.tensorboard.plugins import projector

n_tag = 9
tag_lex = {"O": 0, "B-C": 1, "B-P": 2, "I-C": 3, "I-P": 4, "E-C": 5, "E-P": 6, "S-C": 7, "S-P": 8}


class Ner():
    def __init__(self, sequence_lengths, n_word,
                 embedding_size, lstm_size, batch_size):
        """buids a NER network with an embedding layer, a bidirectionnal LSTM and a CRF layer
        """

        self.x = tf.placeholder(tf.int32, (batch_size, sequence_lengths), name="inputs")
        self.y = tf.placeholder(tf.int32, (batch_size, sequence_lengths), name="labels")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([n_word, embedding_size], -1.0, 1.0),
                name="word_embedding")
            self.config = projector.ProjectorConfig()
            embedding = self.config.embeddings.add()
            embedding.tensor_name = W.name
            self.embedded_wrd = tf.nn.embedding_lookup(W, self.x)
            embedding.metadata_path = "/home/jeremie/PycharmProjects/NER/data_tr/metadata.tsv"
            # produce a [batch_size, sequ_length, embedding_size] shaped tensor

        with tf.name_scope("fw_LSTM_layer"):
            fw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
            fw_cell_d = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=0.5)

        with tf.name_scope("bw_LSTM_layer"):
            bw_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
            bw_cell_d = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=0.5)

        inputs = [tf.squeeze(t) for t in tf.split(self.embedded_wrd, batch_size)]
        biLSTMoutput, __, __ = tf.contrib.rnn.static_bidirectional_rnn(fw_cell_d, bw_cell_d, inputs,
                                                                       dtype=tf.float32)
        with tf.name_scope("linear_layer"):
            # Define Hidden layer weights: 2*n_hidden because of forward + backward cells
            weights = tf.Variable(tf.random_normal([n_tag, 2 * lstm_size]), name="weights")
            tf.summary.histogram("weigth", weights)

            biases = tf.Variable(tf.random_normal([n_tag]), name="biases")
            tf.summary.histogram("biases", biases)

            P = [(tf.transpose(tf.matmul(weights, tf.transpose(X)))) for X in biLSTMoutput]
            # sized[2*lstm_size] and weight [2 * lstm_size, n_tag]
            self.P = tf.stack(P)
            # P shaped [batch_size, sentence_length, n_tag]
            tf.nn.bias_add(P, biases)

        with tf.name_scope("CRF_layer"):
            self.transition_params = tf.Variable(tf.random_normal([n_tag, n_tag]), name="transition_matrix")
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.P, self.y, tf.constant(15 * [sequence_lengths]), transition_params=self.transition_params)

        with tf.name_scope("train"):
            self.loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar('loss', self.loss)
            self.tvars = tf.trainable_variables()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))

        self.merged_summaries = tf.summary.merge_all()


def acc_fn(guess, real):
    # the function aims to find the proportion of interresting words (project/client)
    # which are tagged correctly
    n = len(guess)
    cmp_elt = 0
    cmp_tag = 0
    cmp_tot = 0
    cmp = 0
    cmp_0 = 0
    for i in range(n):
        m = len(guess[i])
        cmp_tot += m
        for j in range(m):
            if (guess[i][j] == real[i][j]) and (real[i][j] == 0):
                cmp_elt += 1
            if (real[i][j] != 0) and (guess[i][j] == real[i][j]):
                cmp_tag += 1
            if (real[i][j] != 0):
                cmp += 1
            if (real[i][j] == 0):
                cmp_0 += 1
    return (float(cmp_elt) / cmp_0, float(cmp_tag) / cmp, cmp / cmp_tot)


def train(args):
    train_input, train_output = rd.getTrData(args.filenames, args.NamedEntities, args.lex)
    sequ_length = 20  # the sequence lengths are meant to be defined in the preprocessing of the data
    nerModel = Ner(sequ_length, args.n_word, args.embedding_size, args.lstm_size, args.batch_size)
    saver = tf.train.Saver(nerModel.tvars)
    with tf.Session() as sess:
        saver.restore(sess, "/home/jeremie/PycharmProjects/NER/data_tr/model.ckpt")
        # tr_writer = tf.summary.FileWriter("/home/jeremie/PycharmProjects/NER/data_tr", sess.graph)
        # sess.run(tf.global_variables_initializer())
        for e in range(args.epoch):
            # TO DO: build a data-pipeline
            for step in range(0, len(train_input), args.batch_size):
                if (step + (args.batch_size) < len(train_input)):
                    sess.run(nerModel.train_op, feed_dict=
                    {nerModel.x: train_input[step:step + (args.batch_size)],
                     nerModel.y: train_output[step:step + (args.batch_size)]})
                    if (step % 28 == 0):
                        sess.run(nerModel.train_op, feed_dict=
                        {nerModel.x: train_input[step:step + (args.batch_size)],
                         nerModel.y: train_output[step:step + (args.batch_size)]})
                        # tr_writer.add_summary(m_summary, step)

                        saver.save(sess, "/home/jeremie/PycharmProjects/NER/data_tr/model.ckpt")

                        P = sess.run(nerModel.P, feed_dict=
                        {nerModel.x: train_input[step:step + (args.batch_size)],
                         nerModel.y: train_output[step:step + (args.batch_size)]})
                        transition_matrix = sess.run(nerModel.transition_params)
                        viterbi, viterbi_scores = [], []
                        for Y in P:
                            tmp_viterbi, tmp_viterbi_scores = tf.contrib.crf.viterbi_decode(np.asarray(Y), np.asarray(
                                transition_matrix))
                            viterbi.append(tmp_viterbi)
                            viterbi_scores.append(tmp_viterbi_scores)
                        # tf.summary.scalar('viterbi_scores', viterbi_scores)

                        acc = acc_fn(viterbi, train_output[step:step + (args.batch_size)])
                        print(acc)

            print("epoch = %d" % e)
            # projector.visualize_embeddings(tr_writer, nerModel.config)

def max_c(viterbi, viterbi_score, vs_max, j, t):
    # find out wich word is tagged as client with the higgest viterbi score
    # to make a prediction
    if (viterbi_score > vs_max):
        i = 0
        while (i<len(viterbi)-1)and (viterbi[i] not in [1, 3, 5, 7]):
            i += 1
        if viterbi[i] in [1,3,5,7]:
            t = 1
            while (i+t < len(viterbi)) and (viterbi[i + t] in [1,3, 5, 7]):
                t += 1
            return (viterbi_score, i, True, t)
    return vs_max, j, False, t

def max_p(viterbi, viterbi_score, vs_max, j, t):
    if (viterbi_score > vs_max):
        i = 0
        while (i<len(viterbi)-1)and (viterbi[i] not in [2,4,6,8]):
            i += 1
        if viterbi[i] in [2,4,6,8]:
            t = 1
            while (i+t < len(viterbi)-1) and (viterbi[i + t] in [2,4,6,8]):
                t += 1
            return (viterbi_score, i, True, t)
    return vs_max, j, False, t

def test(args):
    train_input, train_output = rd.getTrData(args.filenames, args.NamedEntities, args.lex)
    sequ_length = 20
    nerModel = Ner(sequ_length, args.n_word, args.embedding_size, args.lstm_size, args.batch_size)
    saver = tf.train.Saver(nerModel.tvars)
    with tf.Session() as sess:
        saver.restore(sess, "/home/jeremie/PycharmProjects/NER/data_tr/model.ckpt")
        transition_matrix = sess.run(nerModel.transition_params)
        vs_max_c, i_b_c, i_s_c, i_w_c, t_c = 0, 0, 0, 0, 0
        vs_max_p, i_b_p, i_s_p, i_w_p, t_p = 0, 0, 0, 0, 0
        for step in range(0, len(train_input) // args.batch_size):

            P = sess.run(nerModel.P, feed_dict=
                {nerModel.x: train_input[step * (args.batch_size):(step + 1) * (args.batch_size)],
                 nerModel.y: train_output[step * (args.batch_size):(step + 1) * (args.batch_size)]})

            for i in range(len(P)):
                viterbi, viterbi_score = tf.contrib.crf.viterbi_decode(np.asarray(P[i]),
                                                                        np.asarray(transition_matrix))
                vs_max_c, i_w_c, bo_c, t_c = max_c(viterbi, viterbi_score, vs_max_c, i_w_c, t_c)
                vs_max_p, i_w_p, bo_p,t_p = max_p(viterbi, viterbi_score, vs_max_p, i_w_p, t_p)
                if bo_c:
                    i_s_c = i
                    i_b_c = step
                if bo_p:
                    i_s_p = i
                    i_b_p = step
        c = train_input[i_b_c * (args.batch_size) + i_s_c][i_w_c:i_w_c+t_c]
        p = train_input[i_b_p * (args.batch_size) + i_s_p][i_w_p:i_w_p+t_p]
        for key, value in args.lex.iteritems():
            if value in c:
                for i in range(len(c)):
                    if c[i]==value:
                        c[i] = key
            if value in p:
                for i in range(len(p)):
                    if p[i]==value:
                        p[i] = key
        return(c,p)


class Arg():
    def __init__(self, filenames, NamedEntities, embedding_size, lstm_size, batch_size, epoch):
        self.filenames = filenames
        self.NamedEntities = NamedEntities
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.lex = rd.build_vocab(filenames[0])
        for fname in filenames[1:]:
            tmp = rd.build_vocab(fname)
            self.lex = rd.lex_add(self.lex, tmp)
        # rd.write_metadata(self.lex, "/home/jeremie/PycharmProjects/NER/data_tr/metadata.tsv")
        self.n_word = len(self.lex) + 1


tr_args = Arg(["/home/jeremie/PycharmProjects/untitled/fichiertest.txt",
            "/home/jeremie/PycharmProjects/NER/200708006_serveur_Kolab_110Bourgogne.txt"],
           [[tg.NamedEntity(["bretagne", "telecom"], "C"), tg.NamedEntity(["alfresco", "heberge"], "P")],
            [tg.NamedEntity(["110", "bourgogne"], "C"), tg.NamedEntity(["serveur", "kolab"], "P")]],
           25, 50, 15, 200)

test_args = Arg(["/home/jeremie/PycharmProjects/untitled/fichiertest.txt"],
           [[tg.NamedEntity(["bretagne", "telecom"], "C"), tg.NamedEntity(["alfresco", "heberge"], "P")]],
           25, 50, 15, 5)

if __name__ == '__main__':
    print(test(tr_args))
    
