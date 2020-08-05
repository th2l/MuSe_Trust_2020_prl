"""
Created by hvthong
"""
import tensorflow as tf
from configs.configuration import *
import utils
import math


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


class CusLRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, min_lr, lr_start_warmup=0., warmup_steps=10, num_constant=0, T_max=20, num_half_cycle=1.,
                 name=None):
        super(CusLRScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.num_constant = num_constant
        self.T_max = T_max
        self.lr_start_warmup = lr_start_warmup
        self.min_lr = min_lr
        self.num_half_cycle = num_half_cycle
        self.name = name
        pass

    def __call__(self, step):
        with tf.name_scope(self.name or "CusLRScheduler") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype

            min_lr = tf.cast(self.min_lr, dtype)
            lr_start_warmup = tf.cast(self.lr_start_warmup, dtype)

            step_cf = tf.cast(step, dtype)
            wm_steps = tf.cast(self.warmup_steps, dtype=dtype)

            warmup_ratio = tf.where(tf.less_equal(step_cf, wm_steps), step_cf / wm_steps, 0.0)
            use_warmup_lr = tf.where(tf.less_equal(step_cf, wm_steps), 1.0, 0.0)
            warmup_lr = use_warmup_lr * (lr_start_warmup + warmup_ratio * (initial_learning_rate - lr_start_warmup))

            num_constant = tf.cast(self.num_constant, dtype=dtype)

            constant_lr = tf.where(
                tf.logical_and(tf.less_equal(step_cf - wm_steps, num_constant), use_warmup_lr<1),
                initial_learning_rate, 0.0)

            t_max = tf.cast(self.T_max, dtype)
            use_consine_lr = tf.where(tf.logical_and(tf.less_equal(step_cf, t_max), tf.less(wm_steps + num_constant, step_cf)), 1.0, 0.0)
            pi_val = tf.cast(tf.constant(math.pi), dtype)
            num_half_cycle = tf.cast(self.num_half_cycle, dtype)
            cosine_lr = tf.where(use_consine_lr>0., min_lr + (initial_learning_rate - min_lr) * (1 + tf.cos(
                pi_val * num_half_cycle*(step_cf - wm_steps - num_constant) / (t_max - wm_steps - num_constant))) / 2, 0.)

            use_min_lr = tf.where(tf.less_equal(t_max, step_cf), min_lr, 0.0)

            return use_min_lr + cosine_lr + constant_lr + warmup_lr

    def get_config(self):
        ret_config = {'initial_learning_rate': self.initial_learning_rate,
                      'min_lr': self.min_lr,
                      'lr_start_warmup': self.lr_start_warmup,
                      'warmup_steps': self.warmup_steps,
                      'num_constant': self.num_constant,
                      'T_max': self.T_max,
                      'num_half_cycle': self.num_half_cycle,
                      'name': self.name}
        return ret_config


def hardtanh(x, min_val=-1.0, max_val=1.0):
    return tf.keras.backend.clip(x, min_value=min_val, max_value=max_val)


def LstmStack(units, inputs, use_norm=True, layername=None, gaussian_noise=0.):
    if not isinstance(units, (list, tuple)):
        units = [units]

    x = inputs

    for idx in range(len(units)):
        current_name = layername if layername is None else '{}_{}'.format(layername, idx)
        x = tf.keras.layers.LSTM(units=units[idx], return_sequences=True, use_bias=True, name=current_name)(x)
        if use_norm:
            x = tf.keras.layers.LayerNormalization()(x)
        if gaussian_noise > 0:
            x = tf.keras.layers.GaussianNoise(stddev=gaussian_noise)(x)
    return x


def FcStack(units, inputs, act='relu', use_norm=True, layername=None, use_bias=False, gaussian_noise=0.):
    if not isinstance(units, (list, tuple)):
        units = [units]
    x = inputs
    for idx in range(len(units)):
        current_name = layername if layername is None else '{}_{}'.format(layername, idx)
        if len(units) == 1:
            current_name = current_name if layername is None else layername
        x = tf.keras.layers.Dense(units=units[idx], activation='linear', use_bias=use_bias,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  kernel_initializer='he_uniform', name=current_name)(x)
        if use_norm:
            x = tf.keras.layers.BatchNormalization()(x)

        if gaussian_noise > 0:
            x = tf.keras.layers.GaussianNoise(stddev=gaussian_noise)(x)
        if act == 'tanh':
            x = tf.nn.tanh(x)
        elif act == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif act == 'softmax':
            x = tf.nn.softmax(x)
        elif act == 'relu':
            x = tf.nn.relu(x)

    return x


def get_unimodal(hparams, feat_name, use_mask=True, gaussian_noise=0.):
    input_shape = (hparams['seq_length'], FEATURE_NUM[feat_name])
    inputs = tf.keras.Input(shape=input_shape, name=feat_name)

    if use_mask:
        x = tf.keras.layers.Masking(mask_value=0., input_shape=input_shape)(inputs)
    else:
        x = inputs
    seq_enc = LstmStack(hparams['lstm_units'], x, use_norm=True, gaussian_noise=0.)
    seq_fc = FcStack(hparams['fc_units'], seq_enc, act='relu', use_norm=True, gaussian_noise=gaussian_noise)

    return inputs, seq_fc


def get_model(seq_length, feat_names, reg_cls='reg', num_output=2, targets=('arousal', 'valence'), use_mask=True,
              fuse=2, use_multitask_loss=False, gaussian_noise=0.):
    """
    Get un_compiled uni-multi modal
    :param params:
    :param feat_names:
    :param reg_cls:
    :param num_output:
    :param fuse: 1 - early fusion, 2 - late fusion
    :return:
    """
    if fuse != 2:
        raise ValueError('Only support late fusion at this time.')
    assert num_output == len(targets)
    feature_models = dict()
    input_models = dict()
    for fn in feat_names:
        lstm_units = [64, 64, 64]
        fc_units = [32, 32]
        hparams = {'lstm_units': lstm_units, 'fc_units': fc_units, 'seq_length': seq_length}
        input_models[fn], feature_models[fn] = get_unimodal(hparams, fn,
                                                            use_mask=use_mask, gaussian_noise=gaussian_noise)  # Batch size x seq_length x num_feature

    if len(feat_names) > 1:
        n_model = len(feat_names)
        merge_feat = tf.stack(list(feature_models.values()), axis=-1)  # Batch size x seq_length x num_feature x n_model
        merge_wgt = FcStack(units=[n_model], inputs=merge_feat, act='tanh',
                            use_norm=False)  # Batch size x seq_length x num_feature x n_model
        merge_wgt - FcStack(units=[n_model], inputs=merge_wgt, act='sigmoid',
                            use_norm=False)  # Batch size x seq_length x num_feature x n_model

        merge_feat_rescale = tf.keras.layers.multiply(
            [merge_feat, merge_wgt])  # Batch size x seq_length x num_feature x n_model
        last_feat = tf.keras.backend.sum(merge_feat_rescale, axis=-1)  # Batch size x seq_length x num_feature
    else:
        last_feat = list(feature_models.values())[0]

    # Do regression or classification
    if reg_cls == 'reg':
        # Regression
        outs = []
        for idx in range(num_output):
            # outs.append(FcStack([1], inputs=last_feat, act='linear', use_norm=False, layername=targets[idx]))
            cur_out = FcStack([1], inputs=last_feat, act='linear', use_norm=False)
            outs.append(tf.keras.layers.Activation('tanh', name=targets[idx])(cur_out))
        # if 'trustworthiness' in targets and len(targets) > 1:
        #     trust_index = targets.index('trustworthiness')
        #     merge_tasks = tf.concat(outs[:trust_index] + outs[trust_index+1:], axis=-1)
        #     out_trust = FcStack(units=[16], inputs=merge_tasks, act='relu', use_norm=True)
        #     out_trust = FcStack(units=[1], inputs=out_trust, act='linear', use_norm=False, layername='trustworthiness')
        #     outs[trust_index] = out_trust

    else:
        # Classification
        raise ValueError('Un-support classification at this time')
        pass
        # outs = FcStack([num_output], inputs=last_feat, act='softmax', use_norm=False, layername=targets[idx])

    # ret_model = tf.keras.Model(inputs=list(input_models.values()), outputs=out)

    if len(targets) > 1:
        in_out = []
        for idx in range(num_output):
            in_out.append(tf.keras.layers.Input(shape=(seq_length,), name='{}_lb'.format(targets[idx])))
        out_mtl = utils.MultiTaskLoss(num_outputs=num_output, loss_func=utils.ConcordanceLossFunc, trainable=use_multitask_loss)(in_out + outs)

        muse_model = tf.keras.Model(inputs=list(input_models.values()) + in_out, outputs=outs + out_mtl)
    else:
        muse_model = tf.keras.Model(inputs=list(input_models.values()), outputs=outs)
    # muse_model = MuSeModel(inputs=list(input_models.values()), outputs=outs)
    return muse_model
