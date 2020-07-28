"""
Created by hvthong
"""
from comet_ml import Experiment
from comet_ml.exceptions import InterruptedExperiment
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
import tensorflow_addons as tfa
from tqdm import tqdm
import time, gc

def test(cfgs, model, split_name='devel'):
    print('Generate prediction for ', split_name)
    use_multitask_loss = args.use_weight_mtl and len(cfgs.targets) > 1
    test_data = get_data(cfgs, is_training=False, return_infos=True, split_name=split_name,
                         use_multitask_loss=use_multitask_loss)  # Test or devel
    write_out = {'prediction_{}'.format(ky): [] for ky in cfgs.targets if not (ky != 'trustworthiness' and len(cfgs.targets) > 1)}
    write_infos = None

    # ccc_trust = utils.ConcordanceCorrelationCoefficientMetric('val_ccc_trust')
    write_gt = {ky + '_gt': [] for ky in cfgs.targets}

    for sample_batched in tqdm(test_data):
        if split_name != 'test':
            feat_x, feat_y, feat_infos = sample_batched
        else:
            feat_x, feat_infos = sample_batched
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
        for idx in range(len(cfgs.targets)):
            if cfgs.task == 3 and cfgs.targets[idx] != 'trustworthiness':
                continue
            write_out['prediction_{}'.format(cfgs.targets[idx])].append(outputs[idx].numpy().reshape(-1, )[keep_index])
            if split_name != 'test':
                write_gt[cfgs.targets[idx] + '_gt'].append(feat_y[cfgs.targets[idx]].numpy().reshape(-1)[keep_index])

    write_infos_ret = dict()
    for ky in write_infos.keys():
        if ky == 'file_id':
            wky = 'id'
        elif ky == 'timestamp':
            wky = 'timestamp'
        else:
            continue
        write_infos_ret[wky] = np.concatenate(write_infos[ky]).reshape(-1, ).astype(np.int)
        print(wky, write_infos_ret[wky].shape)

    for ky in cfgs.targets:
        if cfgs.task == 3 and len(cfgs.targets) > 1 and ky != 'prediction_{}'.format(ky):
            continue

        write_out['prediction_{}'.format(ky)] = np.concatenate(write_out['prediction_{}'.format(ky)]).reshape(-1, )

        print('prediction_{}'.format(ky), write_out['prediction_{}'.format(ky)].shape)

        if split_name != 'test':
            write_gt[ky + '_gt'] = np.concatenate(write_gt[ky + '_gt']).reshape(-1, )
            print(ky)
            base_code_ccc = utils.CCC(write_out['prediction_{}'.format(ky)], write_gt[ky + '_gt'])
            print('baseline code: ', base_code_ccc)
            ccc_tf = utils.ConcordanceCorrelationCoefficientMetric()
            ccc_tf.update_state(write_gt[ky + '_gt'], write_out['prediction_{}'.format(ky)])
            our_code_cc = ccc_tf.result()
            print('tf: ', our_code_cc)
            if not os.path.isfile('./train_logs/val_ccc_logs.txt'):
                wmode = 'w'
            else:
                wmode = 'a'
            with open('./train_logs/val_ccc_logs.txt', wmode) as logs_fp:
                logs_fp.write('{},{},{},{}\n'.format('_'.join(cfgs.use_data), cfgs.dir, our_code_cc, base_code_ccc))

    write_data = write_infos_ret
    write_data.update(write_out)
    if split_name != 'test':
        write_data.update(write_gt)

    print("Check write data")
    for k, v in write_data.items():
        print(k)
        print(v.shape)
    # for kp, kv in write_data.items():
    #     print(kp)
    #     print(kv.shape)

    df = pd.DataFrame(np.stack(list(write_data.values())).T, columns=list(write_data.keys())).sort_values(
        by=['id', 'timestamp'])
    df['id'] = df['id'].astype(int)
    df['timestamp'] = df['timestamp'].astype(int)
    if args.test != '':
        df.to_csv(os.path.join(cfgs.test, '{}.csv'.format(split_name)), index=False)
    else:
        df.to_csv(os.path.join(cfgs.dir, '{}.csv'.format(split_name)), index=False)


def train(cfgs):
    """

    :param hparams:
    :param use_feat:
    :param task:
    :return:
    """
    experiment = Experiment(project_name="MuSe-Trust-2020", api_key='uG1BcicYOr83KvLjFEZQMrWVg',
                            auto_output_logging='simple', disabled=False)
    # experiment.log_parameters(cfgs)
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/main.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/data_generator.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/utils.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/models.py')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/configs')
    experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder='./src/common.py')

    print('Use feat ', cfgs.use_data)
    use_multitask_loss = args.use_weight_mtl


    METRICS = {ky: utils.ConcordanceCorrelationCoefficientMetric(name='CCC') for ky in cfgs.targets}

    if len(targets) == 1:
        loss_obj = {ky: utils.ConcordanceLoss(name='CCCL') for ky in cfgs.targets}
    else:
        loss_obj = None

    model = models.get_model(cfgs.seq_length, feat_names=cfgs.use_data, reg_cls='reg', num_output=len(cfgs.targets),
                             targets=cfgs.targets, use_mask=cfgs.use_mask, use_multitask_loss=use_multitask_loss, gaussian_noise=cfgs.gaussian_noise)
    # model.summary()

    if cfgs.test == '':
        loaders = {'train': get_data(cfgs, is_training=True, return_infos=False, split_name='train',
                                     use_multitask_loss=len(targets) > 1),
                   'devel': get_data(cfgs, is_training=False, return_infos=False, split_name='devel',
                                     use_multitask_loss=len(targets) > 1)}
        # for smp in loaders['devel']:
        #     print(smp)
        #     continue
        steps_per_epoch = cfgs.steps_per_epoch * (10000 / cfgs.batch_size)# len(list(loaders['train']))
        lr_schedule = models.CusLRScheduler(initial_learning_rate=cfgs.lr_init, min_lr=cfgs.min_lr, lr_start_warmup=0.,
                                            warmup_steps=5 * steps_per_epoch, num_constant=0 * steps_per_epoch,
                                            T_max=cfgs.use_min_lr * steps_per_epoch, num_half_cycle=1)
        if cfgs.opt == 'adam':
            # opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            opt = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

        # tf.keras.utils.plot_model(model, to_file=os.path.join(cfgs.dir, 'model_png.png'), show_shapes=True)
        model.compile(optimizer=opt, loss=loss_obj, weighted_metrics=METRICS)

        monitor_name = 'val_CCC' if len(cfgs.targets) == 1 else 'val_trustworthiness_CCC'

        best_ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(cfgs.dir, 'best_checkpoint.h5'),
                                                       monitor=monitor_name,
                                                       verbose=0, save_best_only=True, save_weights_only=True,
                                                       mode='max')
        verbose_cb = utils.VerboseFitCallBack()
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_name, factor=0.7, patience=3, verbose=0,
        #                                                  mode='max', min_delta=0.0001, min_lr=1e-6)
        # tfb = tf.keras.callbacks.TensorBoard(log_dir=cfgs.dir)

        try:
            # print('Number of steps per epochs: ', len(list(loaders['train'])))
            model.fit(loaders['train'], validation_data=loaders['devel'], steps_per_epoch=steps_per_epoch,
                      epochs=cfgs.epochs, verbose=1, callbacks=[best_ckpt,])#verbose_cb
        except (InterruptedExperiment, KeyboardInterrupt) as exc:
            experiment.log_other("status", str(exc))
            print('Stopped Training')

        experiment.log_model(name=cfgs.dir.split('/')[-1], file_or_folder=cfgs.dir)

        model.load_weights(os.path.join(cfgs.dir, 'best_checkpoint.h5'))

    else:
        if os.path.isdir(cfgs.test):
            ckpt_path = os.path.join(cfgs.test, 'best_checkpoint.h5')
        else:
            ckpt_path = cfgs.test
        if not os.path.isfile(ckpt_path):
            raise ValueError('Cannot load ', ckpt_path)

        model.load_weights(ckpt_path)


    test(cfgs, model, 'devel')
    test(cfgs, model, 'test')
    pass


def get_data(cfgs, is_training=True, return_infos=False, split_name='train', use_multitask_loss=False):
    data_loader = utils.get_split(cfgs.tf_records_folder, is_training=is_training, task=cfgs.task,
                                  split_name=split_name, feature_names=cfgs.use_data, batch_size=cfgs.batch_size,
                                  return_infos=return_infos, targets=cfgs.targets,
                                  use_multitask_loss=use_multitask_loss)
    return data_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MuSe Challenge code')
    parser.add_argument('--dir', type=str, default='./tmp', help='Training logs directory (default: tmp)')
    parser.add_argument('--test', type=str, default='', help='Test checkpoint (default: )')
    parser.add_argument('--opt', type=str, default='adam', help='Optimizer (default: adam)')
    parser.add_argument('--task', type=int, default=3, help='Task for training {1,2,3} (default: 3)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs (default: 30)')
    parser.add_argument('--gaussian_noise', type=float, default=0.01, help='Add gaussian noise to input (default: 0.1)')
    parser.add_argument('--lr_init', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Initial learning rate (default: 1e-5)')
    parser.add_argument('--use_min_lr', type=int, default=20, help='Start to use min lr (default: 20)')
    parser.add_argument('--steps_per_epoch', type=float, default=1.0, help='Steps per epoch (default 1.0)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 8)')
    parser.add_argument('--buffer_size', type=int, default=200,
                        help='Buffer size for shuffle training data (default: 10000)')
    parser.add_argument('--use_mask', action='store_true', help='Use masking in LSTM')
    parser.add_argument('--use_data', nargs='+', help='List of features use in training', required=True)
    parser.add_argument('--use_weight_mtl', action='store_true', help='Use weighted multi-task loss (default: false)')
    parser.add_argument('--multitask', action='store_true',
                        help='Use multitask or not - apply for task 3 only (default: false)')

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

    utils.set_gpu_growth_or_cpu(use_cpu=False)
    utils.set_seed(args.seed)

    cfgs_dict['use_data'] = sorted(cfgs_dict['use_data'])

    targets = utils.get_labels_list(args.task, multitask=args.multitask)
    cfgs_dict['targets'] = targets

    cfgs = dict_to_struct(cfgs_dict)

    print('tfrecords folder: ', cfgs.tf_records_folder)
    hparams = {'num_lstm': 2, 'lstm_units': [64, 16], 'num_fc': 2, 'fc_units': [16, 16], 'SEQ_LENGTH': cfgs.seq_length}
    hparams = dict_to_struct(hparams)

    print(cfgs)

    os.makedirs(cfgs.dir, exist_ok=True)
    train(cfgs)
