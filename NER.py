import tensorflow as tf
import numpy as np
import tagging as tg
import reader as rd
from tensorflow.contrib.tensorboard.plugins import projector

n_tag = 9
tag_lex = {"O":0, "B-C": 1, "B-P":2, "I-C":3, "I-P":4, "E-C":5, "E-P":6, "S-C":7, "S-P":8}


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
            embedding.metadata_path = "/home/jeremie/PycharmProjects/NER/data_tr_d/metadata.tsv"
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

            P = [(tf.transpose(tf.matmul(weights, tf.transpose(Y)))) for Y in biLSTMoutput]
            # sized[2*lstm_size] and weight [2 * lstm_size, n_tag]
            P = tf.stack(P)
            # P shaped [batch_size, sentence_length, n_tag]
            tf.nn.bias_add(P, biases)

        with tf.name_scope("CRF_layer"):
            transition_params = tf.Variable(tf.random_normal([n_tag, n_tag]), name="transition_matrix")
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                P, self.y, tf.constant(15*[sequence_lengths]), transition_params=transition_params)

        with tf.name_scope("prediction"):
            self.prediction = tf.nn.softmax(P)
        with tf.name_scope("accuracy"):
            diff_prediction = tf.equal(tf.argmax(self.prediction, 2), tf.cast(self.y, tf.int64))
            self.accuracy = tf.cast(tf.reduce_mean(tf.cast(diff_prediction, tf.float32)), tf.float32)
            tf.summary.scalar('accuracy', self.accuracy)

        with tf.name_scope("train"):
            self.loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar('loss', self.loss)
            self.tvars = tf.trainable_variables()
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tvars), 10)
            self.train_op = optimizer.apply_gradients(zip(grads, self.tvars))
        self.merged_summaries = tf.summary.merge_all()


def train(args):
    train_input, train_output = rd.getTrData(args.filenames, args.NamedEntities, args.lex)
    sequ_length = 20
    nerModel = Ner(sequ_length, args.n_word,
                   args.embedding_size, args.lstm_size, args.batch_size)
    saver = tf.train.Saver(nerModel.tvars)
    with tf.Session() as sess:
        tr_writer = tf.summary.FileWriter("/home/jeremie/PycharmProjects/NER/data_tr_d", sess.graph)
        # saver.restore(sess, "/home/jeremie/PycharmProjects/NER/data_tr_d/model.ckpt")
        sess.run(tf.global_variables_initializer())
        for e in range(args.epoch):
            for step in range(0, len(train_input), args.batch_size):
                if (step+(args.batch_size) < len(train_input)):
                    sess.run(nerModel.train_op, feed_dict=
                            {nerModel.x : train_input[step:step+(args.batch_size)],
                            nerModel.y : train_output[step:step+(args.batch_size)]})
                    if (step%300 == 0):
                        summary, _ = sess.run([nerModel.merged_summaries, nerModel.train_op], feed_dict=
                                                {nerModel.x: train_input[step:step + (args.batch_size)],
                                                nerModel.y: train_output[step:step + (args.batch_size)]})
                        tr_writer.add_summary(summary, step)
                        saver.save(sess, "/home/jeremie/PycharmProjects/NER/data_tr_d/model.ckpt")
                        # acc = sess.run(nerModel.accuracy, feed_dict=
                        #     {nerModel.x : train_input[:args.batch_size],
                        #     nerModel.y : train_output[:args.batch_size]})
                        # print("accuracy :")
                        # print(acc)
            print("epoch = %d" %e)
        projector.visualize_embeddings(tr_writer, nerModel.config)


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
        # rd.write_metadata(self.lex, "/home/jeremie/PycharmProjects/NER/data_tr_d/metadata.tsv")
        self.n_word = len(self.lex)+1

args = Arg (["/home/jeremie/PycharmProjects/NER/CCTP_MPA_CANADP et annexes techniques.txt",
             "/home/jeremie/PycharmProjects/untitled/fichiertest.txt",
             "/home/jeremie/PycharmProjects/NER/200708006_serveur_Kolab_110Bourgogne.txt"],
             [[tg.NamedEntity(["departement", "de", "paris"], "C"), tg.NamedEntity(["realisation", "d", "une", "application", "i", "net"], "P")],
             [tg.NamedEntity(["bretagne", "telecom"], "C"), tg.NamedEntity(["alfresco", "heberge"], "P")],
             [tg.NamedEntity(["110", "bourgogne"], "C"), tg.NamedEntity(["serveur", "kolab"], "P")]],
             25, 100, 15, 150)

if __name__ == '__main__' :
    train(args)
