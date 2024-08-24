from __future__ import print_function, division
from warnings import warn

from nilmtk.disaggregate import Disaggregator
from keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten, Input, GlobalAveragePooling1D
from keras.layers.pooling import AveragePooling1D
import os
import pandas as pd
import numpy as np
import pickle
from collections import OrderedDict

from tensorflow.keras.optimizers import SGD
from keras.models import Sequential, load_model
from tensorflow.keras.layers import Layer, MultiHeadAttention, LayerNormalization, Embedding
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import random
random.seed(10)
np.random.seed(10)

import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()
BATCH_SIZE_PER_REPLICA = 128
batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class ProbsparseAttention(Layer):
    def __init__(self, sampling_factor=5):
        super(ProbsparseAttention, self).__init__()
        self.sampling_factor = sampling_factor

    def call(self, query_states, key_states, value_states):
        L_K = tf.shape(key_states)[1]
        L_Q = tf.shape(query_states)[1]
        log_L_K = tf.cast(tf.math.ceil(tf.math.log1p(tf.cast(L_K, tf.float32))), tf.int32)
        log_L_Q = tf.cast(tf.math.ceil(tf.math.log1p(tf.cast(L_Q, tf.float32))), tf.int32)

        U_part = tf.minimum(self.sampling_factor * L_Q * log_L_K, L_K)
        index_sample = tf.random.uniform(shape=(U_part,), maxval=L_K, dtype=tf.int32)
        K_sample = tf.gather(key_states, index_sample, axis=1)
        Q_K_sample = tf.matmul(query_states, K_sample, transpose_b=True)

        M = tf.reduce_max(Q_K_sample, axis=-1) - tf.reduce_sum(Q_K_sample, axis=-1) / tf.cast(L_K, tf.float32)
        u = tf.minimum(self.sampling_factor * log_L_Q, L_Q)
        u = tf.minimum(u, tf.shape(M)[-1])  # Ensure u does not exceed the last dimension of M

        M_top = tf.math.top_k(M, k=u, sorted=False).indices

        batch_indices = tf.range(tf.shape(query_states)[0])[:, tf.newaxis]
        batch_indices = tf.tile(batch_indices, [1, u])

        # Ensure compatible shapes for concatenation
        batch_indices = tf.expand_dims(batch_indices, axis=-1)
        M_top = tf.expand_dims(M_top, axis=-1)
        M_top = M_top[:, 0, :, :]

        M_top_indices = tf.concat([batch_indices, M_top], axis=-1)
        
        Q_reduce = tf.gather_nd(query_states, M_top_indices)
        Q_reduce = tf.reshape(Q_reduce, [tf.shape(query_states)[0], u, tf.shape(query_states)[-1]])

        d_k = tf.shape(query_states)[-1]
        attn_scores = tf.matmul(Q_reduce, key_states, transpose_b=True)
        attn_scores = attn_scores / tf.math.sqrt(tf.cast(d_k, tf.float32))
        attn_probs = tf.nn.softmax(attn_scores, axis=-1)
        attn_output = tf.matmul(attn_probs, value_states)

        return attn_output, attn_scores

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sampling_factor': self.sampling_factor,
        })
        return config



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.probsparse_att = ProbsparseAttention()
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output, _ = self.att(inputs, inputs, return_attention_scores=True)
        probsparse_output, _ = self.probsparse_att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        probsparse_output = self.dropout1(probsparse_output, training=training)
        out1 = self.layernorm1(inputs + attn_output + probsparse_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'probsparse_att': self.probsparse_att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
        })
        return config

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
        })
        return config    

class LPpool(Layer):
    def __init__(self, pool_size, strides=None, padding='same'):
        super(LPpool, self).__init__()
        self.avgpool = tf.keras.layers.AveragePooling1D(pool_size, strides, padding)

    def call(self, x):
        x = tf.math.pow(tf.math.abs(x), 2)
        x = self.avgpool(x)
        x = tf.math.pow(x, 1.0 / 2)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'avgpool': self.avgpool,
        })
        return config

class INFORMER(Disaggregator):
    def __init__(self, params):
        self.MODEL_NAME = "INFORMER"
        self.chunk_wise_training = params.get('chunk_wise_training', False)
        self.sequence_length = params.get('sequence_length', 99)
        self.n_epochs = params.get('n_epochs', 10)
        self.models = OrderedDict()
        self.mains_mean = 1800
        self.mains_std = 600
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        print("...............INFORMER partial_fit running...............")
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')
        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))

        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                if len(train_main) > 10:
                    filepath = 'INF-temp-weights-' + str(random.randint(0, 100000)) + '.h5'
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)
                    model.fit(train_x, train_y, validation_data=(v_x, v_y), epochs=self.n_epochs, callbacks=[checkpoint], batch_size=self.batch_size)
                    model.load_weights(filepath)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):
        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length, 1))

            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                prediction = model.predict(test_main_array, batch_size=self.batch_size)

                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]

                prediction = self.appliance_params[appliance]['mean'] + (sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):
        from keras.callbacks import EarlyStopping
        embed_dim = 32
        num_heads = 2
        ff_dim = 32
        vocab_size = 20000
        maxlen = self.sequence_length

        with strategy.scope():
            model = Sequential()
            model.add(Conv1D(16, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
            model.add(LPpool(pool_size=2))

            model.add(TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim))
            model.add(TransformerBlock(embed_dim, num_heads, ff_dim))

            model.add(Flatten())
            model.add(Dropout(0.1))
            model.add(Dense(self.sequence_length))
            model.add(Dropout(0.1))
        model.summary()
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        early_stopping = EarlyStopping(monitor='loss', verbose=1, mode='min')
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                processed_mains_lst.append(pd.DataFrame(new_mains))
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print("Parameters for ", app_name, " were not found!")
                    raise ApplianceNotFoundError()

                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                    new_app_readings = np.array([new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                    new_app_readings = (new_app_readings - app_mean) / app_std
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))

                appliance_list.append((app_name, processed_app_dfs))

            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})

