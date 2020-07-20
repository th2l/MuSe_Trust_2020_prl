"""
Created by hvthong
"""
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, math_ops, state_ops
import tensorflow_probability as tfp
import numpy as np
import os, sys, glob
from functools import partial
import time, random
from pathlib import Path
from configs.configuration import *
from common import *
import copy

def set_gpu_growth_or_cpu(use_cpu=False):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if use_cpu:
            print("Use CPU")
            tf.config.set_visible_devices(gpus[1:], 'GPU')
        else:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs, ", len(logical_gpus), " Logical GPUs")
            except RuntimeError as e:
                print(e)
    print('Use CPU')
    tf.get_logger().setLevel('INFO')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class StreamingCovariance(tf.metrics.Metric):
    # Based on https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/metrics/python/ops/metric_ops.py#L3136-3274
    def __init__(self, name='streaming_covariance'):
        super(StreamingCovariance, self).__init__(name=name)
        self.count_ = self.add_weight("count_", initializer="zeros")
        self.mean_prediction = self.add_weight("mean_prediction", initializer="zeros")
        self.mean_label = self.add_weight("mean_label", initializer="zeros")
        self.comoment = self.add_weight("comoment", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_count = tf.cast(tf.size(y_true), dtype=tf.float32)
        wprediction = y_pred
        wtrue = y_true

        self.count_.assign_add(batch_count)
        prev_count = self.count_ - batch_count  # n_A

        # We update the means by Delta=Error*BatchCount/(BatchCount+PrevCount)
        # batch_mean_prediction is E[x_B] in the update equation
        batch_mean_prediction = tf.math.divide_no_nan(tf.reduce_sum(wprediction), batch_count)
        delta_mean_prediction = tf.math.divide_no_nan((batch_mean_prediction - self.mean_prediction) * batch_count,
                                                      self.count_)

        self.mean_prediction.assign_add(delta_mean_prediction)

        # prev_mean_prediction is E[x_A] in the update equation
        prev_mean_prediction = self.mean_prediction - delta_mean_prediction

        # batch_mean_label is E[y_B] in the update equation
        batch_mean_label = tf.math.divide_no_nan(tf.reduce_sum(wtrue), batch_count)
        delta_mean_label = tf.math.divide_no_nan((batch_mean_label - self.mean_label) * batch_count, self.count_)

        self.mean_label.assign_add(delta_mean_label)
        # prev_mean_label is E[y_A] in the update equation
        prev_mean_label = self.mean_label - delta_mean_label

        unweighted_batch_coresiduals = (y_pred - batch_mean_prediction) * (y_true - batch_mean_label)
        # batch_comoment is C_B in the update equation
        batch_comoment = tf.reduce_sum(unweighted_batch_coresiduals)

        # View delta_comoment as = C_AB - C_A in the update equation above.
        # Since C_A is stored in a var, by how much do we need to increment that var
        # to make the var = C_AB?

        delta_comoment = (batch_comoment + (prev_mean_prediction - batch_mean_prediction) * (
                prev_mean_label - batch_mean_label) * (prev_count * batch_count / self.count_))

        self.comoment.assign_add(delta_comoment)

    def result(self):
        return self.comoment / self.count_

    def reset_states(self):
        self.count_.assign(0.0)
        self.mean_prediction.assign(0.0)
        self.mean_label.assign(0.0)
        self.comoment.assign(0.0)

    def get_config(self):
        return {'name': self.name}

class ConcordanceCorrelationCoefficientMetric(tf.metrics.Metric):
    def __init__(self, name='concordance_correlation_coefficient'):
        super(ConcordanceCorrelationCoefficientMetric, self).__init__(name=name)
        self.cov = StreamingCovariance(name='ccc_cov_')
        self.var_true = StreamingCovariance(name='ccc_var_true')
        self.var_pred = StreamingCovariance(name='ccc_var_pred')
        self.mean_true = tf.metrics.Mean(name='ccc_mean_true')
        self.mean_pred = tf.metrics.Mean(name='ccc_mean_pred')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # if sample_weight is None:
        #     print('Sample weight is None')
        # else:
        #     print('Sample weight is not None')
        batch_count = tf.cast(tf.size(y_true), dtype=tf.float32)
        y_true = tf.reshape(y_true, (batch_count, -1))
        y_pred = tf.reshape(y_pred, (batch_count, -1))
        if sample_weight is not None:
            sample_weight = tf.cast(tf.reshape(sample_weight, (batch_count, -1)), tf.bool)
            y_true = tf.boolean_mask(y_true, sample_weight)
            y_pred = tf.boolean_mask(y_pred, sample_weight)

        self.cov.update_state(y_true, y_pred)
        self.var_true.update_state(y_true, y_true)
        self.var_pred.update_state(y_pred, y_pred)
        self.mean_true.update_state(y_true)
        self.mean_pred.update_state(y_pred)

    def result(self):
        cov_ = self.cov.result()
        var_pred = self.var_pred.result()
        var_true = self.var_true.result()
        mean_pred = self.mean_pred.result()
        mean_true = self.mean_true.result()

        lin_concordance_cc = tf.truediv(2 * cov_, var_pred + var_true + tf.square(mean_pred - mean_true))
        lin_concordance_cc = tf.where(tf.math.is_nan(lin_concordance_cc), -1.0, lin_concordance_cc)
        return lin_concordance_cc

    def reset_states(self):
        self.cov.reset_states()
        self.var_true.reset_states()
        self.var_pred.reset_states()
        self.mean_true.reset_states()
        self.mean_pred.reset_states()

class MSECovarianceLoss(tf.keras.losses.Loss):
    def __init__(self, name='mse_covariance_loss', reduction=tf.keras.losses.Reduction.AUTO):
        self.mse_loss = tf.losses.MeanSquaredError(reduction='sum_over_batch_size')
        self.reduction = reduction
        self.name = name

    def __call__(self, y_true, y_pred, sample_weight=None):
        batch_count = tf.cast(tf.size(y_true), dtype=tf.float32)
        y_true = tf.reshape(y_true, (batch_count, -1))
        y_pred = tf.reshape(y_pred, (batch_count, -1))

        cov_ = tfp.stats.covariance(y_true, y_pred)
        mse_ = self.mse_loss(y_true, y_pred, sample_weight=None)

        current_loss = tf.divide(tf.square(mse_), tf.square(cov_) + 1e-8)
        return tf.reshape(current_loss, [])

    def get_config(self):
        return {'name': self.name, 'reduction': self.reduction}


class ConcordanceLoss(tf.keras.losses.Loss):
    def __init__(self, name='concordance_loss', reduction=tf.keras.losses.Reduction.AUTO):
        self.name = name
        self.reduction = reduction

    def __call__(self, y_true, y_pred, sample_weight=None):
        batch_count = tf.cast(tf.size(y_true), dtype=tf.float32)
        y_true = tf.reshape(y_true, (batch_count, -1))
        y_pred = tf.reshape(y_pred, (batch_count, -1))

        cov_ = tfp.stats.covariance(y_true, y_pred)
        var_pred = tfp.stats.variance(y_pred)
        var_true = tfp.stats.variance(y_true)
        mean_pred = tf.reduce_mean(y_pred)
        mean_true = tf.reduce_mean(y_true)

        ccc_score = tf.truediv(2*cov_, var_pred + var_true + tf.square(mean_pred-mean_true))
        ccc_score = tf.where(tf.math.is_nan(ccc_score), -1.0, ccc_score)   # In range [-1, 1]
        current_loss = 1 - ccc_score     # In range [0, 2]

        return tf.reshape(current_loss, [])

    def get_config(self):
        return {'name': self.name, 'reduction': self.reduction}

def get_labels_list(task, multitask=False):
    if task == 1:
        # Regression
        print("Regression task")
        targets = ["valence", "arousal"]
    elif task == 2:
        # Classification [3 - 3 - 10]
        print("Classification task")
        targets = ["topic", "valence", "arousal"]
    elif task == 3:
        # Regression
        print("Regression task")
        if multitask:
            targets = ["trustworthiness", "valence", "arousal"]
        else:
            targets = ["trustworthiness"]
    else:
        raise ValueError("Unknown task {}".format(task))

    return targets

str_features = tf.io.FixedLenFeature([], tf.string)

parse_dict = {'file_id': str_features, 'chunk_id': str_features,
              'timestamp': str_features, 'au': str_features,
              'deepspectrum': str_features, 'egemaps': str_features,
              'fasttext': str_features, 'gaze': str_features,
              'gocar': str_features, 'landmarks_2d': str_features,
              'landmarks_3d': str_features, 'openpose': str_features, 'pdm': str_features,
              'pose': str_features, 'vggface': str_features,
              'xception': str_features, 'topic': str_features, 'trustworthiness': str_features,
              'valence': str_features, 'arousal': str_features}


def filter_features(x, task, use_feat=None, return_infos=False, targets=()):
    """

    :param x:
    :param task:
    :param use_feat:
    :return:
    """
    must_included = ['arousal', 'valence', 'trustworthiness', 'topic', 'file_id', 'chunk_id', 'timestamp']
    if use_feat is not None:
        must_included = must_included + use_feat

    ret_features = dict()#[]
    ret_labels = dict() #[]
    ret_infos = dict() #[]
    for feat_key in parse_dict.keys():
        if feat_key not in must_included or feat_key not in x.keys():
            continue
        if feat_key in ['file_id', 'chunk_id', 'timestamp']:
            # ret_values[feat_key] = tf.cast(tf.io.parse_tensor(x[feat_key], tf.int64), tf.int32)
            cur_features = tf.cast(tf.io.parse_tensor(x[feat_key], tf.int64), tf.int32)
            ret_infos[feat_key] = cur_features

        elif feat_key in ['arousal', 'valence', 'trustworthiness'] and feat_key in targets:
            if task == 1 or task == 3:
                # ret_values[feat_key] = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
                cur_features = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
            else:
                # ret_values[feat_key] = tf.cast(tf.io.parse_tensor(x[feat_key], tf.int64), tf.int32)
                cur_features = tf.cast(tf.io.parse_tensor(x[feat_key], tf.int64), tf.int32)
            ret_labels[feat_key] = cur_features

        elif feat_key == 'topic':
            # ret_values[feat_key] = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
            cur_features = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
            ret_labels[feat_key] = cur_features
        else:
            # ret_values[feat_key] = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
            cur_features = tf.cast(tf.io.parse_tensor(x[feat_key], tf.double), tf.float32)
            ret_features[feat_key] = cur_features

    sample_mask = tf.logical_and(tf.logical_and(tf.equal(ret_infos['file_id'], 0), tf.equal(ret_infos['chunk_id'], 0)), tf.equal(ret_infos['timestamp'], 0))
    sample_mask = tf.logical_not(sample_mask)

    ret_values = [ret_features]
    if len(ret_labels) > 0:
        ret_values.append(ret_labels)
    if return_infos:
        ret_values.append(ret_infos)
    else:
        ret_values.append(sample_mask)

    return ret_values


def map_dataset(dataset, task, is_test=False):
    """

    :param dataset:
    :param task:
    :param is_test:
    :return:
    """
    current_dict = copy.deepcopy(parse_dict)
    if is_test:
        print('Test data do not contain label => Remove all label key in parse dict')
        key_pop = ['topic', 'trustworthiness', 'arousal', 'valence']
    else:
        key_pop = ['lld']  # We do not use lld at all
        if task == 2:
            key_pop = key_pop + ['trustworthiness']
        elif task == 1:
            key_pop = key_pop + ['topic', 'trustworthiness']
        elif task == 3:
            key_pop = key_pop + ['topic']
        else:
            raise ValueError('Unknown task {}'.format(task))

    for kpop in key_pop:
        _ = current_dict.pop(kpop, None)

    ret_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, features=current_dict),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return ret_dataset, current_dict


def get_split(dataset_dir, is_training, task, split_name, feature_names, batch_size, are_test_labels_available=False, return_infos=False, targets=()):
    """

    :param dataset_dir:
    :param is_training:
    :param task:
    :param split_name:
    :param id_to_partition:
    :param feature_names:
    :param batch_size:
    :param seq_length:
    :param buffer_size:
    :param are_test_labels_available:
    :return:
    """
    assert len(targets) > 0, 'Oh no, number of targets should be larger than 0'
    root_path = Path(os.path.join(dataset_dir, split_name))
    # List of tfrecords for current split (train, devel or test)
    paths = sorted([str(x) for x in root_path.glob('*.tfrecords')])
    print('Number of tfrecords {}: '.format(split_name), len(paths))

    if is_training:
        dataset = tf.data.Dataset.from_tensor_slices(paths).cache().shuffle(buffer_size=len(paths)*5, seed=1, reshuffle_each_iteration=True)
        dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=tf.data.experimental.AUTOTUNE,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = tf.data.TFRecordDataset(paths).cache()

    is_test = not (split_name != 'test' or are_test_labels_available)

    dataset, feat_dict = map_dataset(dataset, task=task, is_test=is_test)

    dataset = dataset.map(lambda x: filter_features(x, task, use_feat=feature_names, return_infos=return_infos, targets=targets),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)#.prefetch(tf.data.experimental.AUTOTUNE)
    # Batching
    dataset = dataset.batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def CCC(X1, X2):
    x_mean = np.nanmean(X1)
    y_mean = np.nanmean(X2)
    x_var = 1.0 / (len(X1) - 1) * np.nansum((X1 - x_mean) ** 2)
    y_var = 1.0 / (len(X2) - 1) * np.nansum((X2 - y_mean) ** 2)

    covariance = np.nanmean((X1 - x_mean) * (X2 - y_mean))
    return round((2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2),4)