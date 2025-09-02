import ast
import json
import os
import re

import numpy as np
import tensorflow as tf
from scipy.spatial import distance

print(f"Using {__name__}")


# sliding window function
# this is the same function that was applied to the dataset during preprocessing
@tf.function(reduce_retracing=True)
def sliding_window(tensor, window_size, step):
    # tensor = tf.cast(tensor, tf.float32)
    frames = tf.signal.frame(tensor, window_size, step, axis=-1)
    return frames


# min-max scaling normalization
@tf.function
def fast_minmax_scale(x, axis):
    x_min = tf.reduce_min(x, axis=axis, keepdims=True)
    x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    x_scaled = (x - x_min) / (x_max - x_min)
    return x_scaled


# formatting number function
# (maybe) useful for accessing the subject folders and the specific files
def format_num(num, digits=3):
    num_string = str(num)
    while len(num_string) < digits:
        num_string = "0" + num_string
    return num_string


# gets the absolute paths of all files in a folder
def get_absolute_paths(folder_path):
    absolute_paths = [
        os.path.join(folder_path, item) for item in os.listdir(folder_path)
    ]
    return sorted(absolute_paths)


def get_label(sample_path, n):
    # filenames are of format "SXXX_sampleYYYY.npy"
    # therefore, we split the string twice, once on 'S' and once on the '_'
    num = os.path.basename(sample_path).split("S")[1].split("_")[0]
    label = tf.one_hot(int(num), n, dtype=tf.int8)
    return label


def get_label2(sample_path, n):
    # filenames are of format "SXXXRYY.edf"
    # therefore, we split the string twice, once on 'S' and once on the 'R'
    num = os.path.basename(sample_path).split("S")[1].split("R")[0]
    label = tf.one_hot(int(num), n, dtype=tf.int8)
    return label


def checkpoint_exists(dir, chosen_channels):
    try:
        model_paths = get_absolute_paths(dir)
        regex = f"{chosen_channels}"
        regex = regex.replace("[", "\[")
        regex = regex.replace("]", "\]")
        p = re.compile(regex)

        arr = list(map(lambda x: p.findall(x) != [], model_paths))
        return any(arr)
    except Exception as e:
        print(f"Error: Empty dir, check {dir}\n{e}")
        return None


def get_latest_epoch(dir, chosen_channels):
    try:
        model_paths = get_absolute_paths(dir)

        regex = f"(?<={chosen_channels}-epoch_)\d+"
        regex = regex.replace("[", "\[")
        regex = regex.replace("]", "\]")
        p = re.compile(regex)

        def get_epoch(x):
            if p.findall(x) == []:
                return -1
            return int(p.findall(x)[0])

        epoch_nums = list(map(lambda x: get_epoch(x), model_paths))
        latest_epoch = max(epoch_nums)
        latest_epoch_model = np.argmax(epoch_nums)

        return latest_epoch, model_paths[latest_epoch_model]
    except Exception as e:
        print(f"Error: Something went wrong. Check {dir}\n{e}")
        return None, None


# this needs to run after every orthogonalization
def create_samples_and_labels(data, Gamma, D):
    # data = tf.convert_to_tensor(data, tf.float32)
    data_strided = sliding_window(data, Gamma, D)
    data_strided = np.array(data_strided)

    num_of_subjects = data_strided.shape[0]
    num_of_samples = data_strided.shape[2]
    num_of_data = num_of_subjects * num_of_samples

    data_strided = np.transpose(data_strided, [0, 2, 1, 3])
    data_strided = np.reshape(
        data_strided, (num_of_data, data_strided.shape[2], data_strided.shape[3])
    )
    out_data = tf.convert_to_tensor(data_strided, dtype=tf.float32)

    out_labels = tf.TensorArray(tf.int8, size=num_of_data, dynamic_size=True)
    count = 0
    for subject in range(num_of_subjects):
        for _ in range(num_of_samples):
            sample_label = tf.one_hot(subject, num_of_subjects, dtype=tf.int8)
            out_labels = out_labels.write(count, sample_label)
            count += 1
    out_labels = out_labels.stack()

    return out_data, out_labels


def read_acc_file(acc_save_file):
    checked_channels = set()
    best_channels = []
    
    print("Looking for save file...")
    if not os.path.exists(acc_save_file):
        print("Save file not found")
        return [], []
    print("Save file found!")
    
    print("Reading save file for already checked channels and found best channels...")
    with open(acc_save_file, "r") as f:
        for line in f:
            # Each line format: "[channels] : stats"
            match_channels_and_stats = re.match(r"\[(.*?)\]\s*:\s*\{(.*?)\}", line)
            match_best_channels = re.match(r"Best channel\(s\) :\s*\[(.*?)\]", line)
            
            # ignore lines that don't match with either regex
            if not match_channels_and_stats and not match_best_channels:
                continue
            
            channels_str = match_channels_and_stats.group(1) if match_channels_and_stats else match_best_channels.group(1)
            try:
                channels = ast.literal_eval(f"[{channels_str}]") # yes I know it's unsafe
                
                # if we reach a best channels line, that means we've exhausted the search space once
                # therefore, clear checked channels set
                if match_best_channels:
                    best_channels = channels
                    checked_channels.clear()
                
                for c in channels:
                    checked_channels.add(c)

            except Exception as e:
                print(e)
                continue
    
    print(f"Found {len(checked_channels)} checked channels")
    print(f"Best channels found: {best_channels}")
    return best_channels, checked_channels


def orthogonalize(data, k, best_channels, V):
    tik = data[:, k : k + 1, :].copy()
    for j, _ in enumerate(best_channels):
        projection = list(
            map(
                lambda a, b: np.dot(a, b.T) / np.dot(b, b.T),
                tik,
                V[:, j : j + 1, :],
            )
        )
        # projection = np.dot(tik, V[:,j:j+1,:]) / np.dot(V[:,j:j+1,:], V[:,j:j+1,:])
        tik = tik - projection * V[:, j : j + 1, :]
    return tik


# Method for orthogonalizing channels one by one in order
def orthogonalize_channels(data, channels):
    """Orthogonalize channels one by one in order.

    Args:
        data (NDArray): the input data array
        channels (list): the ordered list of channels to orthogonalize

    Returns:
        NDArray: the orthogonalized data
    """
    first_channel = channels[0]
    tmp = data[:, first_channel : first_channel + 1, :].copy()
    for idx, channel in enumerate(channels):
        if idx == 0:
            continue
        orth_tmp = orthogonalize(
            data=data, k=channel, best_channels=[first_channel], V=tmp
        )
        tmp = np.concatenate([tmp, orth_tmp], axis=1)
    return tmp


def orthogonalize_data(data):
    """Orthogonalize any given subset of data.

    Args:
        data (NDArray): the entire subset of data we want to orthogonalize (assuming channels are already chosen)

    Returns:
        NDArray: the same subset of data but orthogonalized
    """
    V = data[:, 0:1, :].copy()
    for i in range(1, data.shape[1]):
        orth_tmp = orthogonalize(data=data, k=i, best_channels=[0], V=V)
        V = np.concatenate([V, orth_tmp], axis=1)
    return V


def euclidean_distance(A, B):
    dist = distance.euclidean(A, B)
    return dist


def cosine_distance(A, B):
    dist = distance.cosine(A, B)
    return dist


def manhattan_distance(A, B):
    dist = distance.cityblock(A, B)
    return dist


def create_dataset(data, labels, sliding_window_out, strategy):
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int8)
    with strategy.scope():
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        dataset = dataset.map(sliding_window_out, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(64, drop_remainder=True)
        # dataset = dataset.prefetch(64)
    return dataset


if __name__ == "__main__":
    print("Devices Available: ", tf.config.list_physical_devices())
