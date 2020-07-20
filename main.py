"""
Created by hvthong
"""
from comet_ml import Experiment
import os, random, argparse
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
# from configs import trust_cfgs, wild_cfgs, topic_cfgs
from configs.configuration import *
from common import *
from tqdm import tqdm
import utils
import models


def test(cfgs, model):
    test_data = get_data(cfgs, is_training=False, return_infos=True, split_name='devel')  # Test or devel
    write_out = {ky: [] for ky in cfgs.targets}
    write_infos = None

    # ccc_trust = utils.ConcordanceCorrelationCoefficientMetric('val_ccc_trust')
    write_gt = {ky + '_gt': [] for ky in cfgs.targets}

    for sample_batched in test_data:
        feat_x, feat_y, feat_infos = sample_batched
        outputs = model(feat_x)

        if write_infos is None:
            write_infos = {ky: [kv.numpy().reshape(-1, )] for ky, kv in feat_infos.items()}
        else:
            for ky, kv in feat_infos.items():
                write_infos[ky].append(kv.numpy().reshape(-1, ))

        # Remove padded rows
        keep_index = None
        for ky in write_infos.keys():
            if keep_index is None:
                keep_index = write_infos[ky][-1] == 0.0
            else:
                keep_index = np.logical_and(keep_index, write_infos[ky][-1] == 0)

        keep_index = np.logical_not(keep_index)
        for ky in write_infos.keys():
            write_infos[ky][-1] = write_infos[ky][-1][keep_index]

        if len(cfgs.targets) == 1:
            outputs = [outputs]
        for idx in range(len(outputs)):
            write_out[cfgs.targets[idx]].append(outputs[idx].numpy().reshape(-1, )[keep_index])
            write_gt[cfgs.targets[idx] + '_gt'].append(feat_y[cfgs.targets[idx]].numpy().reshape(-1)[keep_index])

    for ky in write_infos.keys():
        write_infos[ky] = np.concatenate(write_infos[ky]).reshape(-1, )

    for ky in cfgs.targets:
        write_out[ky] = np.concatenate(write_out[ky]).reshape(-1, )
        write_gt[ky + '_gt'] = np.concatenate(write_gt[ky + '_gt']).reshape(-1, )
        print(ky)
        print('baseline code: ', utils.CCC(write_out[ky], write_gt[ky + '_gt']))
        ccc_tf = utils.ConcordanceCorrelationCoefficientMetric()
        ccc_tf.update_state(write_gt[ky + '_gt'], write_out[ky])
        print('tf: ', ccc_tf.result())

    write_data = write_infos
    write_data.update(write_out)
    write_data.update(write_gt)

    # for kp, kv in write_data.items():
    #     print(kp)
    #     print(kv.shape)

    pd.DataFrame(np.stack(list(write_data.values())).T, columns=list(write_data.keys())).sort_values(
        by=['file_id', 'timestamp']).to_csv('devel.csv', index=False)

    pass


def train(cfgs):
    """

    :param hparams:
    :param use_feat:
    :param task:
    :return:
    """
    experiment = Experiment(project_name="MuSe-Trust-2020", api_key='uG1BcicYOr83KvLjFEZQMrWVg', auto_output_logging='simple')
    # experiment.log_parameters(cfgs)
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='main.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='data_generator.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='utils.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='models.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='configs')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='main.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='common.py')

    if cfgs.task == 1:
        num_output = 2
    elif cfgs.task == 3:
        num_output = 3
    else:
        raise ValueError('Un-support task {} at this time'.format(cfgs.task))
    print('Use feat ', cfgs.use_data)

    loaders = {'train': get_data(cfgs, is_training=True, return_infos=False, split_name='train'),
               'devel': get_data(cfgs, is_training=False, return_infos=False, split_name='devel')}

    METRICS = {ky: utils.ConcordanceCorrelationCoefficientMetric(name='CCC') for ky in cfgs.targets}

    loss_obj = {ky: utils.ConcordanceLoss(name='CCCL') for ky in cfgs.targets}
    opt = tf.keras.optimizers.Adam(learning_rate=cfgs.lr_init)

    model = models.get_model(cfgs.seq_length, feat_names=cfgs.use_data, reg_cls='reg', num_output=len(cfgs.targets),
                             targets=cfgs.targets, use_mask=cfgs.use_mask)

    model.compile(optimizer=opt, loss=loss_obj, weighted_metrics=METRICS)

    monitor_name = 'val_CCC' if len(cfgs.targets) == 1 else 'val_trustworthiness_CCC'

    best_ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(cfgs.dir, 'best_checkpoint.h5'), monitor=monitor_name,
                                                   verbose=0, save_best_only=True, save_weights_only=True, mode='max')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_name, factor=0.1, patience=3, verbose=0, mode='max', min_delta=0.0001, min_lr=1e-6)
    # tfb = tf.keras.callbacks.TensorBoard(log_dir=cfgs.dir)

    model.fit(loaders['train'], validation_data=loaders['devel'], epochs=cfgs.epochs, verbose=2, callbacks=[best_ckpt, reduce_lr])

    model.load_weights(os.path.join(cfgs.dir, 'best_checkpoint.h5'))

    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder=cfgs.dir)
    # test(cfgs, model)
    pass


def get_data(cfgs, is_training=True, return_infos=False, split_name='train'):
    data_loader = utils.get_split(cfgs.tf_records_folder, is_training=is_training, task=cfgs.task,
                                  split_name=split_name, feature_names=cfgs.use_data, batch_size=cfgs.batch_size,
                                  return_infos=return_infos, targets=cfgs.targets)
    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuSe Challenge code')
    parser.add_argument('--dir', type=str, default='tmp', help='Training logs directory (default: tmp)')
    parser.add_argument('--task', type=int, default=3, help='Task for training {1,2,3} (default: 3)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs (default: 30)')
    parser.add_argument('--gaussian_noise', type=float, default=0.1, help='Add gaussian noise to input (default: 0.1)')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate (default: 0.1)')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 8)')
    parser.add_argument('--buffer_size', type=int, default=200,
                        help='Buffer size for shuffle training data (default: 10000)')
    parser.add_argument('--use_mask', action='store_true', help='Use masking in LSTM')

    args = parser.parse_args()

    print("TensorFlow version: ", tf.__version__)
    cfgs_dict = vars(args)
    cfgs_dict['challenge_folder'] = TASK_FOLDER[map_tasks['task{}'.format(cfgs_dict['task'])]]
    cfgs_dict['tf_records_folder'] = TF_RECORDS_FOLDER[map_tasks['task{}'.format(cfgs_dict['task'])]]
    cfgs_dict['output_folder'] = OUTPUT_FOLDER[map_tasks['task{}'.format(cfgs_dict['task'])]]
    cfgs_dict['seq_length'] = SEQ_LENGTH

    use_data = {"au": False, "deepspectrum": False, "egemaps": False, "fasttext": False, "gaze": False, "gocar": False,
                "landmarks_2d": False, "landmarks_3d": False, "lld": False, "openpose": False, "pdm": False,
                "pose": False, "vggface": False, "xception": False}

    if cfgs_dict['task'] in [1, 3]:
        use_data.update({'deepspectrum': True, 'fasttext': True, 'vggface': True, "landmarks_2d": True})
    elif cfgs_dict['task'] == 2:
        use_data.update({'deepspectrum': True, 'fasttext': True, 'vggface': True})
    else:
        raise ValueError("Unknown task {}".format(cfgs_dict['task']))

    utils.set_gpu_growth_or_cpu(use_cpu=False)
    utils.set_seed(args.seed)

    cfgs_dict['use_data'] = [k for k, v in use_data.items() if v is True]

    targets = utils.get_labels_list(args.task, multitask=False)
    cfgs_dict['targets'] = targets

    cfgs = dict_to_struct(cfgs_dict)

    loaders = get_data(cfgs=cfgs)

    print('tfrecords folder: ', cfgs.tf_records_folder)
    hparams = {'num_lstm': 2, 'lstm_units': [64, 16], 'num_fc': 2, 'fc_units': [16, 16], 'SEQ_LENGTH': cfgs.seq_length}
    hparams = dict_to_struct(hparams)

    print(cfgs)
    # for feat, label in loaders['train']:
    #     print(feat.keys(), label.keys())
    #     idx += 1
    #     if idx > 20:
    #         break

    train(cfgs)
