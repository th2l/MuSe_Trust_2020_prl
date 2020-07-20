"""
Created by hvthong
"""
import tensorflow as tf
from configs.configuration import *
import utils

class MuSeModel(tf.keras.Model):
    def train_step(self, data):
        feat_x, feat_y, feat_mask = data

        with tf.GradientTape() as tape:
            y_pred = self(feat_x, training=True)
            loss = self.compiled_loss(feat_y, y_pred, sample_weight=feat_mask, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weight
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update metrics
        self.compiled_metrics.update_state(y_true=feat_y, y_pred=y_pred, sample_weight=feat_mask)
        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        feat_x, feat_y, feat_mask = data
        # Compute predictions
        y_pred = self(feat_x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(feat_y, y_pred, sample_weight=feat_mask, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(feat_y, y_pred, sample_weight=feat_mask)
        # Return a dict mapping metric names to current value.
        return {m.name: m.result() for m in self.metrics}

def LstmStack(units, inputs, use_norm=True, layername=None):
    if not isinstance(units, (list, tuple)):
        units = [units]

    x = inputs

    for idx in range(len(units)):
        current_name = layername if layername is None else '{}_{}'.format(layername, idx)
        x = tf.keras.layers.LSTM(units=units[idx], return_sequences=True, use_bias=True, name=current_name)(x)
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)
    return x

def FcStack(units, inputs, act='relu', use_norm=True, layername=None):
    if not isinstance(units, (list, tuple)):
        units = [units]
    x = inputs
    for idx in range(len(units)):
        current_name = layername if layername is None else '{}_{}'.format(layername, idx)
        if len(units) == 1:
            current_name = current_name if layername is None else layername
        x = tf.keras.layers.Dense(units=units[idx], activation='linear', use_bias=True,
                                        kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                        kernel_initializer='he_uniform', name=current_name)(x)
        if use_norm:
            x = tf.keras.layers.BatchNormalization()(x)

        if act == 'tanh':
            x = tf.nn.tanh(x)
        elif act == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif act == 'softmax':
            x = tf.nn.softmax(x)
        elif act == 'relu':
            x = tf.nn.relu(x)

    return x


def get_unimodal(hparams, feat_name, use_mask=True):
    input_shape = (hparams['seq_length'], FEATURE_NUM[feat_name])
    inputs = tf.keras.Input(shape=input_shape, name=feat_name)

    if use_mask:
        x = tf.keras.layers.Masking(mask_value=0., input_shape=input_shape)(inputs)
    else:
        x = inputs
    seq_enc = LstmStack(hparams['lstm_units'], x, use_norm=True)
    seq_fc = FcStack(hparams['fc_units'], seq_enc, act='relu', use_norm=True)

    return inputs, seq_fc

def get_model(seq_length, feat_names, reg_cls='reg', num_output=2, targets=('arousal', 'valence'), use_mask=True):
    """
    Get un_compiled uni-multi modal
    :param params:
    :param feat_names:
    :param reg_cls:
    :param num_output:
    :return:
    """
    assert num_output == len(targets)
    feature_models = dict()
    input_models = dict()
    for fn in feat_names:
        lstm_units = [max(FEATURE_NUM[fn]//4, 16), max(FEATURE_NUM[fn]//16, 16)]
        fc_units = [max(FEATURE_NUM[fn]//16, 16), max(FEATURE_NUM[fn]//16, 8), 8]
        hparams = {'lstm_units': lstm_units, 'fc_units': fc_units, 'seq_length': seq_length}
        input_models[fn], feature_models[fn] = get_unimodal(hparams, fn, use_mask=use_mask)   # Batch size x seq_length x num_feature

    if len(feat_names) > 1:
        n_model = len(feat_names)
        merge_feat = tf.stack(list(feature_models.values()), axis=-1)    # Batch size x seq_length x num_feature x n_model
        merge_wgt = FcStack(units=[n_model], inputs=merge_feat, act='tanh', use_norm=False) # Batch size x seq_length x num_feature x n_model
        merge_wgt - FcStack(units=[n_model], inputs=merge_wgt, act='tanh', use_norm=False) # Batch size x seq_length x num_feature x n_model

        merge_feat_rescale = tf.keras.layers.multiply([merge_feat, merge_wgt])  # Batch size x seq_length x num_feature x n_model
        last_feat = tf.keras.backend.sum(merge_feat_rescale, axis=-1)    # Batch size x seq_length x num_feature
    else:
        last_feat = feat_names.values()[0]

    # Do regression or classification
    if reg_cls == 'reg':
        # Regression
        outs = []
        for idx in range(num_output):
            outs.append(FcStack([1], inputs=last_feat, act='linear', use_norm=False, layername=targets[idx]))
    else:
        # Classification
        outs = FcStack([num_output], inputs=last_feat, act='softmax', use_norm=False, layername=targets[idx])

    # ret_model = tf.keras.Model(inputs=list(input_models.values()), outputs=out)

    muse_model = tf.keras.Model(inputs=list(input_models.values()), outputs=outs)
    # muse_model = MuSeModel(inputs=list(input_models.values()), outputs=outs)
    return muse_model


