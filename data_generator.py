import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import collections

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.signal as spsig

from common import make_dirs_safe
from configs.configuration import *
import sys
from tqdm import tqdm
import librosa as lb

AUTOTUNE = tf.data.experimental.AUTOTUNE
features_name_tf = []

def read_task_2_class_label_file(task_folder, target, segment_id):
    sequence_list = list()

    path = task_folder + "/label_segments" + "/" + target + "/" + repr(segment_id) + ".csv"
    df = pd.read_csv(path, delimiter=",")

    chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1,)))))

    for chunk_id in chunk_ids:
        sequence = df.loc[df['segment_id'] == chunk_id]
        sequence = sequence["class_id"].values.reshape((1, 1))[0, 0]

        sequence = np.float32(sequence)

        s = 0
        if target == "topic":
            subsequence = np.zeros((10,), dtype=np.float32)
            subsequence[int(sequence)] = 1.0
        elif target in ["arousal", "valence"]:
            subsequence = np.zeros((3,), dtype=np.float32)
            subsequence[int(sequence)] = 1.0
        else:
            raise ValueError

        sequence_list.append((s, chunk_id, subsequence))

    return sequence_list


def read_sequential_label_file(task_folder, target, segment_id):
    sequence_list = list()

    path = task_folder + "/label_segments" + "/" + target + "/" + repr(segment_id) + ".csv"
    df = pd.read_csv(path, delimiter=",")

    chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1,)))))

    for chunk_id in chunk_ids:
        sequence = df.loc[df['segment_id'] == chunk_id]
        sequence = sequence["value"].values.reshape((sequence.shape[0], 1)).astype(np.float32)

        num_sub_seq = sequence.size // SEQ_LEN
        if sequence.size % SEQ_LEN > 0:
            num_sub_seq += 1

        for s in range(num_sub_seq):
            start_step = s * SEQ_LEN
            end_step = (s + 1) * SEQ_LEN
            if s == num_sub_seq - 1:
                subsequence = sequence[start_step:]
            else:
                subsequence = sequence[start_step:end_step]
            sequence_list.append((s, chunk_id, subsequence))

    return sequence_list


def read_feature_file(task_folder, segment_id, feature_name_list):
    feature_list = collections.defaultdict(list)

    feature_len_dict = dict()

    for feature_name in feature_name_list:
        if feature_name == "lld":
            path = task_folder + "/feature_segments/unaligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        else:
            path = task_folder + "/feature_segments/fasttext_aligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        try:
            df = pd.read_csv(path, delimiter=",")
        except:
            print('File not found. ', path)
            sys.exit(0)

        chunk_ids = sorted(set(list(df["segment_id"].values.reshape((-1,)))))

        feature_len_dict[feature_name] = dict()

        for chunk_id in chunk_ids:
            feature = df.loc[df['segment_id'] == chunk_id]
            feature = feature.values[:, 2:]
            feature = feature.reshape(feature.shape).astype(np.float32)

            feature_len_dict[feature_name][chunk_id] = feature.shape[0]

            if feature_name == "lld":
                feature = spsig.resample_poly(feature, up=1, down=25, axis=0)

                try:
                    egemaps_shape = feature_len_dict["egemaps"][chunk_id]
                except KeyError:
                    continue

                if feature.shape[0] > egemaps_shape:
                    feature = feature[:egemaps_shape, :]
                elif feature.shape[0] < egemaps_shape:
                    feature = np.vstack([feature,
                                         feature[-1, :] * np.ones((egemaps_shape - feature.shape[0],
                                                                   feature.shape[1]),
                                                                  dtype=feature.dtype)])
                feature = feature.astype(np.float32)

            s = 0
            subsequence = feature
            if subsequence.shape[0] > 500:
                subsequence = subsequence[:500, :]

            feature_list[feature_name].append((s, chunk_id, subsequence))

    return feature_list

def read_label_data(task_folder, segment_id, label_name_list):
    """

    :param task_folder:
    :param segment_id:
    :param label_name:
    :return:
    """
    label_dict = dict()
    timestamp_dict = dict()
    chunk_dict = dict()
    for label_name in label_name_list:
        path = task_folder + "/label_segments/" + label_name + "/" + repr(segment_id) + ".csv"
        try:
            df = pd.read_csv(path, delimiter=',', header=0)
        except:
            print('File not found. ', path)
            sys.exit(0)
        label_dict[label_name] = df["value"].values.reshape(-1, 1).astype(np.float)
        timestamp_dict[label_name] = df["timestamp"].values.reshape(-1, 1).astype(np.int)
        chunk_dict[label_name] = df["segment_id"].values.reshape(-1, 1).astype(np.int)

    return chunk_dict, timestamp_dict, label_dict

def read_processed_data(task_folder, segment_id, feature_name_list, task, get_label=True):

    feature_dict = dict()
    timestamp_dict = dict()
    chunk_id_dict = dict()

    for feature_name in feature_name_list:
        if feature_name == "lld" or feature_name == "raw_audio":
            # print('Our system do not support lld')
            continue
        else:
            path = task_folder + "/feature_segments/label_aligned/" + feature_name + "/" + repr(segment_id) + ".csv"
        try:
            df = pd.read_csv(path, delimiter=",", header=0)
        except:
            print('File not found. ', path)
            sys.exit(0)

        chunk_id_dict[feature_name] = df["segment_id"].values.reshape(-1,1).astype(np.int)
        timestamp_dict[feature_name] = df["timestamp"].values.reshape(-1, 1).astype(np.int)
        feature_dict[feature_name] = df.values[:, 2:].astype(np.float)

        assert feature_dict[feature_name].shape[1] == FEATURE_NUM[feature_name], "Oh no, feature {} do not match shape {}".format(feature_name, FEATURE_NUM[feature_name])

    # Check consistent between timestamp and segment_id of features
    chunk_check = dict()
    chunk_check.update(chunk_id_dict)

    timestamp_check = dict()
    timestamp_check.update(timestamp_dict)

    if task == 'c1_muse_wild':
        label_name_list = ['valence', 'arousal']
    elif task == 'c3_muse_trust':
        label_name_list = ['trustworthiness', 'valence', 'arousal']
    else:
        raise ValueError('Do not support task {}'.format(task))

    if get_label:
        label_chunk_dict, label_timestamp_dict, label_dict = read_label_data(task_folder, segment_id, label_name_list)
        chunk_check.update(label_chunk_dict)
        timestamp_check.update(label_timestamp_dict)
    else:
        label_dict = dict()
        for label_name in label_name_list:
            label_dict[label_name] = None

    chunk_time_ = {'chunk_id': None, "timestamp": None}
    num_samples = 0
    for ky in chunk_check.keys():
        if chunk_time_['chunk_id'] is None:
            chunk_time_['chunk_id'] = chunk_check[ky]
            chunk_time_['timestamp'] = timestamp_check[ky]
            num_samples = chunk_id_dict[ky].shape[0]
        else:
            res = np.array_equal(chunk_time_['chunk_id'], chunk_check[ky]) and np.array_equal(chunk_time_['timestamp'], timestamp_check[ky])
            if not res:
                raise ValueError('Inconsistent data in file {}'.format(segment_id))

    id_dict = segment_id * np.ones(num_samples, dtype=np.int).reshape(-1, 1)

    # Get raw audio features
    chunk_id_unq = np.unique(chunk_time_['chunk_id'], return_counts=False)
    raw_audio_feat = []
    SR = 8000
    segment_len = 0.25 * SR
    for cid_index in range(len(chunk_id_unq)):
        current_cid = chunk_id_unq[cid_index]
        current_timestamps = chunk_time_['timestamp'][chunk_time_['chunk_id'] == current_cid]

        raw_waveform, sr = lb.load(os.path.join(task_folder, 'audio_segments', str(segment_id), '{}_{}.wav'.format(segment_id, current_cid)), sr=SR)
        duration = current_timestamps[-1] - current_timestamps[0]
        num_segments = int((duration / 1000*sr) /segment_len) + 1
        segments = np.linspace(0, duration/1000*sr, num=num_segments, dtype=int)
        if num_segments != len(current_timestamps):
            print('stop here')
        for idx in range(num_segments-1):
            start = segments[idx]
            stop = segments[idx+1]
            current_segm = raw_waveform[start: stop]
            if len(current_segm) < int(segment_len):
                current_segm = np.pad(current_segm, (0, int(segment_len)-len(current_segm)), mode='constant', constant_values=0)
            raw_audio_feat.append(current_segm)

        last_segment = raw_waveform[segments[-1]:]
        if len(last_segment) < segment_len:
            last_segment = np.pad(last_segment, (0, int(segment_len)-len(last_segment)), mode='constant', constant_values=0)
        else:
            last_segment = last_segment[:int(segment_len)]
        if len(last_segment) != int(segment_len):
            print('Error here ', segment_id, current_cid)
            sys.exit(0)
        raw_audio_feat.append(last_segment)

    try:
        raw_audio_feat = np.array(raw_audio_feat).astype(np.float)
    except:
        print('Error in here ', segment_id)
        sys.exit(0)
    if raw_audio_feat.shape[0] != chunk_time_['chunk_id'].shape[0]:
        print('Need to check Error here ', segment_id)
    feature_dict['raw_audio'] = raw_audio_feat
    return id_dict, chunk_time_['chunk_id'], chunk_time_['timestamp'], feature_dict, label_dict

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _parse_function(element):
    ints_features = tf.io.FixedLenFeature([], tf.int64)
    floats_features = tf.io.FixedLenFeature([], tf.float32)
    float_seq_features = tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
    str_features = tf.io.FixedLenFeature([], tf.string)

    parse_dict = {'file_id': str_features, 'chunk_id': str_features,
                  'timestamp': str_features, 'au': str_features,
                  'deepspectrum': str_features, 'egemaps': str_features,
                  'fasttext': str_features, 'gaze': str_features,
                  'gocar': str_features, 'landmarks_2d': str_features,
                  'landmarks_3d': str_features,
                  'openpose': str_features, 'pdm': str_features,
                  'pose': str_features, 'vggface': str_features,
                  'xception': str_features, }
                  # 'topic': str_features, 'trustworthiness': str_features,
                  # 'arousal': str_features, 'valence': str_features}

    example_message = tf.io.parse_single_example(element, parse_dict)

    b_feature = tf.io.parse_tensor(example_message['file_id'], out_type=tf.int64)  # get byte string
    au_feature = tf.io.parse_tensor(example_message['au'], out_type=tf.double)  # get byte string
    print(b_feature)
    return b_feature

def serialize_example(*arg):
    feature_str = dict()

    for idx in range(len(features_name_tf)):
        serialized_np = tf.io.serialize_tensor(arg[idx])
        feature_str[features_name_tf[idx]] = _bytes_feature(serialized_np)
        # if features_name_tf[idx] in ['file_id', 'chunk_id', 'timestamp']:
        #     feature_str[features_name_tf[idx]] = _bytes_feature(arg[idx])
        # else:
        #     feature_str[features_name_tf[idx]] = _bytes_feature(arg[idx])

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_str))
    return example_proto.SerializeToString()

def tf_serialize_example(feature_dict):
    v_list = []
    for k, v in feature_dict.items():
        v_list.append(v)

    tf_string = tf.py_function(serialize_example, v_list, tf.string)
    return tf.reshape(tf_string, ())

def serialize_sample(task_name, write_folder, task_folder, sample_id, feature_names, partition,
                     are_test_labels_available=False):
    global features_name_tf
    try:
        get_label = partition != 'test' or are_test_labels_available
        # Read data files
        if task_name in ["c1_muse_wild", "c3_muse_trust"]:
            file_id, chunk_ids, timestamps, features, targets = read_processed_data(task_folder, sample_id, feature_names, task=task_name, get_label=get_label)

        elif task_name == "c2_muse_topic":
            feature_names = [n for n in feature_names if n != "lld"]
            features = read_feature_file(task_folder, sample_id, feature_names)
        else:
            raise ValueError

        features_list = {'file_id': file_id, 'chunk_id': chunk_ids, 'timestamp': timestamps}

        if partition != 'test' or are_test_labels_available:
            features_list.update(targets)

        features_list.update(features)
        features_name_tf = list(features_list.keys())

        # Divide chunks into sub-segment
        features_list_tf = dict()
        chunk_id_uniques = np.unique(chunk_ids)
        current_samples = 0

        for xn in features_name_tf:
            features_list_tf[xn] = []

        for cid in chunk_id_uniques:
            index_mask = (chunk_ids == cid).flatten()
            timestamp_vals = timestamps[index_mask].flatten()
            timestamp_index = np.argsort(timestamp_vals)
            num_frames = len(timestamp_index)
            num_sub_segments = num_frames // SEQ_LENGTH

            if num_frames % SEQ_LENGTH > 0:
                num_sub_segments = num_sub_segments + 1
            current_samples = current_samples + num_sub_segments * SEQ_LENGTH

            for feat_name in features_name_tf:
                current_feat = features_list[feat_name]
                if current_feat.ndim == 1 or current_feat.shape[0] == 1:
                    current_feat = current_feat.reshape(-1, 1)
                current_feat = current_feat[index_mask, :]

                current_feat_sort = current_feat[timestamp_index, :]

                for idx in range(num_sub_segments):
                    if idx * SEQ_LENGTH > num_frames:
                        break
                    ed_index = (idx + 1) * SEQ_LENGTH
                    if ed_index <= num_frames:
                        current_sub = current_feat_sort[idx * SEQ_LENGTH: ed_index, :]
                    else:
                        current_sub = current_feat_sort[idx * SEQ_LENGTH: num_frames, :]
                        current_sub = np.pad(current_sub, pad_width=((0, SEQ_LENGTH-(num_frames%SEQ_LENGTH)), (0, 0)), mode='constant', constant_values=0)

                    features_list_tf[feat_name].append(current_sub)

                # try:
                #     features_list_tf[feat_name] = np.stack(features_list_tf[feat_name])
                # except:
                #     for x in features_list_tf[feat_name]:
                #         print(x.shape, sample_id)
                #
                #     sys.exit(0)

        for feat_name in features_name_tf:
            features_list_tf[feat_name] = np.stack(features_list_tf[feat_name])

        features_dataset = tf.data.Dataset.from_tensor_slices(features_list_tf)
        serialized_features_dataset = features_dataset.map(tf_serialize_example, num_parallel_calls=AUTOTUNE)

        #with tf.io.TFRecordWriter(write_folder + "/{}.tfrecords".format(sample_id)) as writer:
        writer = tf.data.experimental.TFRecordWriter(write_folder + "/{}.tfrecords".format(sample_id))
        # writer = tf.data.experimental.TFRecordWriter( "abc.tfrecords".format(sample_id))
        writer.write(serialized_features_dataset)
        #
        #
        # raw_dataset = tf.data.TFRecordDataset(['/mnt/hvthong/Project/MuSe/dataset/c3_muse_trust/tfrecords/devel/15.tfrecords'])
        # parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
        # for raw_record in parsed_dataset.take(2):
        #     print(repr(raw_record))
        #     print("OK")
        # for raw_record in raw_dataset.take(2):
        #     tmp = _parse_function(raw_record)
        #     continue
        return current_samples


    except FileNotFoundError:
        print("File not found:", sample_id)
        return

    pass

def get_partition(partition_proposal_path, name="third"):
    df = pd.read_csv(partition_proposal_path, delimiter=",")
    data = df[["Id", "Proposal"]].values

    id_to_partition = dict()
    partition_to_id = collections.defaultdict(set)

    for i in range(data.shape[0]):
        sample_id = int(data[i, 0])
        partition = data[i, 1]

        id_to_partition[sample_id] = partition
        partition_to_id[partition].add(sample_id)

    return id_to_partition, partition_to_id


def main(task_name, partition_proposal_path, tf_records_folder, task_folder, feature_names, are_test_labels_available=False):
    make_dirs_safe(tf_records_folder)

    id_to_partition, partition_to_id = get_partition(partition_proposal_path)

    for partition in partition_to_id.keys():
        if partition == 'train':
            continue
        print("Making tfrecords for", partition, "partition.")
        current_tf_records_folder = tf_records_folder + '/' + partition
        make_dirs_safe(current_tf_records_folder)
        total_samples = 0
        for sample_id in tqdm(partition_to_id[partition]):
            m = serialize_sample(task_name, current_tf_records_folder, task_folder, sample_id, feature_names, partition, are_test_labels_available)
            total_samples += m
        print('Number of samples: ', total_samples)


if __name__ == "__main__":
    # TASK_NAME = "c2_muse_topic"
    # main(task_name=TASK_NAME,
    #      partition_proposal_path=PARTITION_PROPOSAL_PATH,
    #      tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
    #      task_folder=TASK_FOLDER[TASK_NAME],
    #      feature_names=FEATURE_NAMES,
    #      are_test_labels_available=False)
    #
    print("SEQUENCE LENGTH: ", SEQ_LENGTH)
    TASK_NAME = "c3_muse_trust"
    print(TASK_NAME)
    main(task_name=TASK_NAME,
         partition_proposal_path=PARTITION_PROPOSAL_PATH,
         tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
         task_folder=TASK_FOLDER[TASK_NAME],
         feature_names=FEATURE_NAMES,
         are_test_labels_available=False)

    # TASK_NAME = "c1_muse_wild"
    # print(TASK_NAME)
    # main(task_name=TASK_NAME,
    #      partition_proposal_path=PARTITION_PROPOSAL_PATH,
    #      tf_records_folder=TF_RECORDS_FOLDER[TASK_NAME],
    #      task_folder=TASK_FOLDER[TASK_NAME],
    #      feature_names=FEATURE_NAMES,
    #      are_test_labels_available=False)
