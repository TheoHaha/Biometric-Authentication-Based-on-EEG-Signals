import ast
import os
import re
from itertools import permutations
from typing import Callable

import mne
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

import proposed_cnn
import resnet18
from my_utils import (
    create_samples_and_labels,
    fast_minmax_scale,
    get_absolute_paths,
    orthogonalize,
    orthogonalize_data,
    sliding_window,
)
from user_authentication import EEGAuthenticator

## GLOBAL PARAMETERS
# sliding window parameters
T = 160
delta = 4
h = 20
C = 64

# sampling window parameters
Gamma = (h - 1) * delta + T
D = 8

# number of subjects used in training
n = 90

# paths
dataset_path = "dataset/"
saves_path = "saves2/"


def load_data(dataset_path, start=0, end=n):
    data_paths = get_absolute_paths(dataset_path)

    print(f"Found {len(data_paths)} subjects in the dataset.")

    data_for_use = data_paths[start:end]

    # load raw data to an array
    raw_data = []
    for sample in data_for_use:
        raw_signal = mne.io.read_raw_edf(sample, verbose=False, preload=True)
        raw_data.append(raw_signal.get_data())

    # the signals aren't consistent in length so we cut the longer ones
    max_cols = 9600
    raw_data = np.array([signal[:, :max_cols] for signal in raw_data])

    # apply minmax scaling normalization
    data = []
    for subject in raw_data:
        normalized = fast_minmax_scale(subject, axis=0)
        data.append(normalized)
    data = np.array(data)

    return data


def prepare_and_save_user_data(
    alpha_users_n: int,
    chosen_channels: list,
    strategy: tf.distribute.Strategy,
    alpha_test_size=0.2,
    orthogonalize=True,
    save_dir="user_data/",
) -> None:
    """Prepares and saves alpha and beta user data.

    Args:
        alpha_users_n (int): How many alpha users we have.
        chosen_channels (list): Which channels were chosen.
        strategy (tf.distribute.Strategy): What strategy is going to be used for the calculations.
        alpha_test_size (float, optional): How much of the alpha user subset will be used for validation during training. Defaults to 0.2.
        orthogonalize (bool, optional): Whether to orthogonalize the signals. Defaults to True.
    """
    saved_args = {**locals()}
    with open(save_dir + "args.txt", "w+") as f:
        f.write(repr(saved_args))

    def save_user_data(data, name):
        np.save(save_dir + f"{name}.npy", data)

    with strategy.scope():
        alpha_users = load_data(dataset_path=dataset_path, end=alpha_users_n)[
            :, chosen_channels, :
        ]
        if orthogonalize:
            alpha_users = orthogonalize_data(alpha_users)

        alpha_X, alpha_y = create_samples_and_labels(alpha_users, Gamma, D)
        alpha_train_X, alpha_test_X, alpha_train_y, alpha_test_y = train_test_split(
            alpha_X.numpy(),
            alpha_y.numpy(),
            test_size=alpha_test_size,
            shuffle=False,
        )
        save_user_data(alpha_train_X, "alpha_train_X")
        save_user_data(alpha_test_X, "alpha_test_X")
        save_user_data(alpha_train_y, "alpha_train_y")
        save_user_data(alpha_test_y, "alpha_test_y")

        beta_users = load_data(dataset_path=dataset_path, start=alpha_users_n, end=109)[
            :, chosen_channels, :
        ]
        if orthogonalize:
            beta_users = orthogonalize_data(beta_users)
        beta_X, beta_y = create_samples_and_labels(beta_users, Gamma, D)
        save_user_data(beta_X, "beta_X")
        save_user_data(beta_y, "beta_y")
    print(f"Data saved in {save_dir}")


# TODO: implement or depreciate
def preprocess_and_save_dataset(
    strategy: tf.distribute.Strategy,
    test_size=0.2,
    orthogonalize=True,
    save_dir="dataset_preprocessed/",
) -> None:
    saved_args = {**locals()}
    with open(save_dir + "args.txt", "w+") as f:
        f.write(repr(saved_args))

    def save_user_data(data, name):
        np.save(save_dir + f"{name}.npy", data)

    with strategy.scope():
        data = load_data(dataset_path=dataset_path, end=108)

        if orthogonalize:
            data = orthogonalize_data(data)

        data_X, data_y = create_samples_and_labels(data, Gamma, D)
        data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(
            data_X.numpy(),
            data_y.numpy(),
            test_size=test_size,
            shuffle=False,
        )
        save_user_data(data_train_X, "data_train_X")
        save_user_data(data_test_X, "data_test_X")
        save_user_data(data_train_y, "data_train_y")
        save_user_data(data_test_y, "data_test_y")

    print(f"Data saved in {save_dir}")


# TODO: implement this
def load_accuracy_data(accuracy_data_path):
    regex = r"\[.+\]"
    acc_file = open(accuracy_data_path, "r")

    acc_data_str = acc_file.read()
    acc_file.close()

    print(re.findall(regex, acc_data_str))


# sliding window routine
@tf.function
def sliding_window_out(arr, label):
    out = sliding_window(arr, T, delta)
    out = tf.transpose(out, perm=[1, 2, 0])
    out = tf.ensure_shape(out, (h, T, arr.shape[0]))
    return out, label


def train_CNN_model(data, search_space, best_channels, strategy, acc_save_file) -> None:
    # setting up orthogonal forward search for channels
    # symbols as given in the algorithm
    # symbol: E
    # search_space = [25, 26, 28, 29]  # [c for c in range(20,C)]#[10,40,61]
    # symbol: C
    # best_channels = [22, 23, 24, 27]
    # symbol: V
    V = None
    # run the orthogonalization mechanism one time, in case we resume training with >1 best_channels
    if len(best_channels) > 1:
        print(f"Continuing training with channels {best_channels}")
        first_channel = best_channels[0]
        V = data[:, first_channel : first_channel + 1, :].copy()
        for idx, channel in enumerate(best_channels):
            if idx == 0:
                continue
            orth_tmp = orthogonalize(
                data=data, k=channel, best_channels=[first_channel], V=V
            )
            V = np.concatenate([V, orth_tmp], axis=1)
    elif len(best_channels) == 1:
        V = data[:, best_channels, :].copy()
    else:
        V = []
    # symbol: i
    i = 0

    # orthogonal forward search
    acc = {0: 0}

    tf.keras.backend.clear_session()
    while True:  # max(acc.values()) < 0.99:
        acc = {}
        tik_all = {}

        # for every channel k in the E
        print(f"Search space: {search_space}")
        for k in search_space:
            accuracy_on_K = 0
            V_tmp = []
            chosen_channels = []
            # if i = 1
            if len(best_channels) == 0:
                chosen_channels = [k]
                # tik = u_k
                tik = data[:, chosen_channels, :].copy()
                V_tmp = tik
            else:
                chosen_channels = best_channels.copy()
                chosen_channels.append(k)

                tik = orthogonalize(data, k, best_channels, V)

                V_tmp = np.concatenate([V, tik], axis=1)
                # tik = gram_schmidt(data[:, chosen_channels, :])
            # pdb.set_trace()
            V_tmp_data, V_tmp_labels = create_samples_and_labels(V_tmp, Gamma, D)
            # print("V_data shape:", V_data.shape)
            # print("V_labels shape:", V_labels.shape)
            # pdb.set_trace()
            print(f"Training on channel(s) {chosen_channels}")
            accuracy_on_K = proposed_cnn.train_model_on_V(
                V_tmp_data,
                V_tmp_labels,
                chosen_channels,
                saves_path,
                h,
                T,
                sliding_window_out,
                strategy,
            )
            print(f"Accuracy on channel(s) {chosen_channels}: {accuracy_on_K}")

            f = open(acc_save_file, "a")
            f.write(f"{chosen_channels} : {accuracy_on_K}\n")
            f.close()

            acc[k] = accuracy_on_K
            tik_all[k] = tik

        values = np.array(list(acc.values()))
        keys = np.array(list(acc.keys()))
        k_star = int(keys[np.argmax(values)])

        search_space.remove(k_star)
        best_channels.append(k_star)
        print(f"Best channels: {best_channels}")
        print("==================================================")

        V = (
            tik_all[k_star]
            if len(best_channels) == 0
            else np.concatenate([V, tik_all[k_star]], axis=1)
        )
        i += 1

        # TODO: better save & load
        if len(best_channels) == 3:
            return

        if len(search_space) == 0:
            break


# TODO: Complete this and run it
def train_resnet18_model(
    data, search_space, best_channels, strategy, acc_save_file="saves/accuracies_resnet18.txt"
) -> None:
    # Detect already checked permutations
    checked_permutations = set()
    print("Checking for already checked permutations...")
    for fname in os.listdir(saves_path):
        if fname.startswith("resnet18_C_") and "-epoch_30" in fname:
            # Extract the channel list from the filename
            start = fname.find("C_") + 2
            end = fname.find("-epoch_30")
            channels_str = fname[start:end]
            try:
                channels = ast.literal_eval(channels_str)
                checked_permutations.add(tuple(channels))
            except Exception as e:
                print(e)
                continue
    print(f"Found {len(checked_permutations)} checked perms")
    
    # orthogonalize channels one by one
    def orthogonalize_channels(channels):
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

    search_space_P = [
        p for p in permutations(search_space, 3) if tuple(p) not in checked_permutations
    ]
    print(f"Search space permutations: {len(search_space_P)}")
    # return
    # symbol: i
    i = 0

    acc = {0: 0}

    tf.keras.backend.clear_session()
    while True:
        acc = {}
        
        # for every throuple of channels k in the search space
        for k in search_space_P:
            accuracy_on_K = 0
            V_tmp = []
            chosen_channels = list(k)

            V_tmp = orthogonalize_channels(k)
            V_tmp_data, V_tmp_labels = create_samples_and_labels(V_tmp, Gamma, D)
            print(f"Training on channels {chosen_channels}")
            accuracy_on_K = resnet18.train_model_on_V(
                V_tmp_data,
                V_tmp_labels,
                chosen_channels,
                saves_path,
                h,
                T,
                sliding_window_out,
                strategy,
            )
            print(f"Accuracy on channels {chosen_channels}: {accuracy_on_K}")

            f = open(acc_save_file, "a")
            f.write(f"{chosen_channels} : {accuracy_on_K}\n")
            f.close()

            acc[k] = accuracy_on_K
            # tik_all[k] = tik

        values = np.array(list(acc.values()))
        keys = np.array(list(acc.keys()))
        k_star = int(keys[np.argmax(values)])

        search_space.remove(k_star)
        best_channels.append(k_star)
        print("==================================================")
        i += 1

        if max(values) >= 0.999:
            print(f"Stopping training, max accuracy reached: {max(values)} with channels {acc[k_star]}")
            break


def full_train_resnet18_model(data, strategy) -> None:
    # with strategy.scope():
    data = orthogonalize_data(data)
    # model = resnet18.create_model(data.shape[0])
    # print(data.shape)
    samples, labels = create_samples_and_labels(data, Gamma, D)
    print(samples.shape)
    train_X, test_X, train_y, test_y = train_test_split(
        samples.numpy(), labels.numpy(), test_size=0.2, shuffle=False
    )
    print(train_X.shape)

    # dataset_train = create_dataset(train_X, train_y, sliding_window_out, strategy)
    # print([e for e in dataset_train.take(1)])
    # dataset_test = create_dataset(test_X, test_y, sliding_window_out, strategy)

    # stats = model.fit(
    #     x=dataset_train,
    #     validation_data=dataset_test,
    #     epochs=30,
    #     steps_per_epoch=None,
    #     validation_steps=None,
    #     verbose=1,
    # )
    print("==============================================")
    print("Job done")
    # print(stats)


def calculate_and_save_thresholds(authenticator: EEGAuthenticator, test_X, test_y):
    with open("saves/thresholds.txt", "a") as f:
        eucl_threshold = authenticator.calculate_threshold(
            distance.euclidean, test_X=test_X, test_y=test_y
        )
        f.write(f"euclidean threshold: {eucl_threshold}\n")

    with open("saves/thresholds.txt", "a") as f:
        cos_threshold = authenticator.calculate_threshold(
            distance.cosine, test_X=test_X, test_y=test_y
        )
        f.write(f"cosine threshold: {cos_threshold}\n")

    with open("saves/thresholds.txt", "a") as f:
        manh_threshold = authenticator.calculate_threshold(
            distance.cityblock, test_X=test_X, test_y=test_y
        )
        f.write(f"manhattan threshold: {manh_threshold}\n")


# TODO: implement
def login(
    user_i: int,  # basically username
    user_data_in: np.ndarray | int,  # basically password
    authenticator: EEGAuthenticator,
    distance_fn: Callable,
    threshold: float,
):
    A = user_data_in
    S = user_i
    _, logged_in = authenticator.authenticate_user(A, S, distance_fn, threshold)
    if logged_in:
        print(f"User {user_i} authenticated")
    else:
        print(f"User {user_i} not authenticated")


def get_all_test_user_samples(user_i: int, test_X, test_y):
    test_labels = np.argmax(test_y, axis=1)
    samples_to_keep = []
    for i, label in enumerate(test_labels):
        if label == user_i:
            samples_to_keep.append(i)
    return test_X[samples_to_keep, :, :]


def main() -> None:
    strategy = tf.distribute.get_strategy()
    data = load_data(dataset_path=dataset_path, end=10)
    
    train_resnet18_model(
        data=data,
        search_space=list(range(22, 30)),
        best_channels=None,
        strategy=strategy,
    )
    # chosen_channels = [22, 23, 24]

    # prepare_and_save_user_data(
    #     alpha_users_n=90, alpha_test_size=0.2, chosen_channels=chosen_channels, strategy=strategy
    # )

    # print(f"Data shape: {beta_users.shape}")

    # acc_save_file = os.path.join(saves_path, "accuracies.txt")  # noqa: F841
    # model_dir = "saves/C_[22, 23, 24]-epoch_30.keras"
    # model = tf.keras.models.load_model(model_dir)
    # alpha_train_X = np.load("user_data/alpha_train_X.npy")
    # alpha_train_y = np.load("user_data/alpha_train_y.npy")

    # alpha_test_X = np.load("user_data/alpha_test_X.npy")
    # alpha_test_y = np.load("user_data/alpha_test_y.npy")

    # full_train_resnet18_model(data=load_data(dataset_path, start=0, end=108), strategy=strategy)
    # print(f"{alpha_test_X.shape}, {alpha_test_y.shape}")

    # authenticator = EEGAuthenticator(
    #     user_data=alpha_train_X,
    #     user_labels=alpha_train_y,
    #     model=model,
    #     Gamma=Gamma,
    #     D=D,
    #     T=T,
    #     delta=delta,
    #     strategy=strategy,
    #     random_state=42,
    # )

    # authenticator.calculate_threshold(distance.cosine, alpha_test_X, alpha_test_y)

    # authenticator.calculate_threshold(distance_fn=distance.cosine, test_X=alpha_test_X, test_y=alpha_test_y)
    # authenticator.calculate_cdist(0,1,"cosine")
    # calculate_and_save_thresholds(authenticator)
    # print(authenticator.authenticate_user(41, 41, distance.cosine, 0.5))
    # calculate_and_save_thresholds(authenticator)


if __name__ == "__main__":
    main()
