import ast
import os
import re
from datetime import datetime
from itertools import permutations
from typing import Callable

import keras
import mne
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping  # noqa: F401
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

import proposed_cnn
import proposed_resnet18
from my_utils import (
    calculate_model_score_from_stats,
    create_dataset,
    create_samples_and_labels,
    fast_minmax_scale,
    get_absolute_paths,
    orthogonalize,
    orthogonalize_channels,
    orthogonalize_data,
    read_acc_file,
    sliding_window,
)
from user_authentication import EEGAuthenticator
import matplotlib.pyplot as plt

keras.mixed_precision.set_global_policy("mixed_float16")

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
saves_path = "saves2"


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
        strategy (keras.distribute.Strategy): What strategy is going to be used for the calculations.
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
        alpha_users = alpha_users.astype(np.float32)
        if orthogonalize:
            alpha_users = orthogonalize_data(alpha_users)

        alpha_X, alpha_y = create_samples_and_labels(alpha_users, Gamma, D)
        alpha_train_X, alpha_test_X, alpha_train_y, alpha_test_y = train_test_split(
            alpha_X.numpy(),
            alpha_y.numpy(),
            test_size=alpha_test_size,
            shuffle=True,
        )
        save_user_data(alpha_train_X, "alpha_train_X")
        save_user_data(alpha_test_X, "alpha_test_X")
        save_user_data(alpha_train_y, "alpha_train_y")
        save_user_data(alpha_test_y, "alpha_test_y")

        beta_users = load_data(dataset_path=dataset_path, start=alpha_users_n, end=108)[
            :, chosen_channels, :
        ]
        beta_users = beta_users.astype(np.float32)
        if orthogonalize:
            beta_users = orthogonalize_data(beta_users)
        beta_X, beta_y = create_samples_and_labels(beta_users, Gamma, D)
        save_user_data(beta_X, "beta_X")
        save_user_data(beta_y, "beta_y")
    print(f"Data saved in {save_dir}")


# sliding window routine
def sliding_window_out(arr, label):
    out = sliding_window(arr, T, delta)
    out = tf.transpose(out, (1, 2, 0))
    # No keras equivalent for tf.ensure_shape, so skip or use assert
    assert out.shape == (h, T, arr.shape[0])
    return out, label


def channel_selection_CNN(data, search_space, strategy, acc_save_file) -> None:
    # setting up orthogonal forward search for channels
    # symbols as given in the algorithm
    # symbol: E
    # search_space = [25, 26, 28, 29]  # [c for c in range(20,C)]#[10,40,61]
    # symbol: C
    # best_channels = [22, 23, 24, 27]
    # symbol: V
    V = None
    # Detect already checked channels from acc_save_file
    best_channels, checked_channels, scores_loaded = read_acc_file(acc_save_file)

    # remove checked_channels from search_space
    search_space_ = set(search_space)
    search_space_ = search_space_.difference(checked_channels)
    search_space_ = search_space_.difference(best_channels)

    # symbol: i
    i = 0
    scores = {0: 0} if not scores_loaded else scores_loaded.copy()

    # after loading the file, check if search space is empty and if the best channels
    # haven't been recorded
    if len(search_space_) == 0 and len(best_channels) + len(checked_channels) == len(
        search_space
    ):
        k_star = max(scores_loaded, key=scores_loaded.get)  # if scores_loaded else None

        search_space_ = set(search_space)
        search_space_.remove(k_star)
        best_channels.append(k_star)

        with open(acc_save_file, "a") as f:
            f.write(f"Best channel(s) : {best_channels}\n")

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

    # after loading the file, ensure tik_all includes all the checked channels
    tik_all = {}
    if len(checked_channels) > 0 and len(checked_channels) < len(search_space_):
        for k in checked_channels:
            tik = orthogonalize(data, k, best_channels, V)
            tik_all[k] = tik

    # orthogonal forward search
    while len(best_channels) < 3:  # and max(acc.values()) < 0.9999:
        if not scores_loaded:
            best_chan = None
            best_score = 0.0
        else:
            best_chan = max(scores_loaded, key=scores_loaded.get)
            best_score = scores_loaded[best_chan]
            scores_loaded.clear()

        # for every channel k in the E
        print(f"Search space: {search_space_}")
        for k in search_space_:
            keras.backend.clear_session()

            stats_on_K = 0
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
            stats_on_K = proposed_cnn.train_model_on_V(
                V_tmp_data,
                V_tmp_labels,
                chosen_channels,
                saves_path,
                h,
                T,
                sliding_window_out,
                strategy,
            )
            # stats_on_K = proposed_resnet18.train_model_on_V(
            #     V_tmp_data,
            #     V_tmp_labels,
            #     chosen_channels,
            #     saves_path,
            #     h,
            #     T,
            #     sliding_window_out,
            #     strategy,
            # )
            print(f"Stats on channel(s) {chosen_channels} : {stats_on_K}")

            with open(acc_save_file, "a") as f:
                f.write(f"{chosen_channels} : {stats_on_K}\n")

            score = calculate_model_score_from_stats(stats_on_K)
            if score > best_score:
                best_score = score
                best_chan = k

            tik_all[k] = tik
            print(f"Score on channel(s) {chosen_channels}: {score}")
            print("-------------------------------------------------")

        # values = np.array(list(scores.values()))
        # keys = np.array(list(scores.keys()))
        # k_star = int(keys[np.argmax(values)])

        # scores.clear()

        search_space_.discard(best_chan)
        best_channels.append(best_chan)
        print(f"Best channel(s): {best_channels}")
        with open(acc_save_file, "a") as f:
            f.write(f"Best channel(s) : {best_channels}\n")
        print("==================================================")

        V = (
            tik_all[best_chan]
            if len(best_channels) == 1
            else np.concatenate([V, tik_all[best_chan]], axis=1)
        )
        tik_all.clear()
        i += 1


def channel_selection_resnet18(data, search_space, strategy, acc_save_file) -> None:
    # setting up orthogonal forward search for channels
    # symbols as given in the algorithm
    # symbol: E
    # search_space = [25, 26, 28, 29]  # [c for c in range(20,C)]#[10,40,61]
    # symbol: C
    # best_channels = [22, 23, 24, 27]
    # symbol: V
    V = None
    # Detect already checked channels from acc_save_file
    best_channels, checked_channels, scores_loaded = read_acc_file(acc_save_file)

    # remove checked_channels from search_space
    search_space_ = set(search_space)
    search_space_ = search_space_.difference(checked_channels)
    search_space_ = search_space_.difference(best_channels)

    # symbol: i
    i = 0
    # scores = {0: 0} if not scores_loaded else scores_loaded.copy()

    # after loading the file, check if search space is empty and if the best channels
    # haven't been recorded
    if len(search_space_) == 0 and len(best_channels) + len(checked_channels) == len(
        search_space
    ):
        # values = np.array(list(scores.values()))
        # keys = np.array(list(scores.keys()))
        # k_star = int(keys[np.argmax(values)])
        k_star = max(scores_loaded, key=scores_loaded.get)  # if scores_loaded else None
        # scores.clear()
        # scores = {0: 0}

        search_space_ = set(search_space)
        search_space_.remove(k_star)
        best_channels.append(k_star)

        with open(acc_save_file, "a") as f:
            f.write(f"Best channel(s) : {best_channels}\n")

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

    # after loading the file, ensure tik_all includes all the checked channels
    tik_all = {}
    if len(checked_channels) > 0 and len(checked_channels) < len(search_space_):
        for k in checked_channels:
            tik = orthogonalize(data, k, best_channels, V)
            tik_all[k] = tik

    # orthogonal forward search
    while len(best_channels) < 3:  # and max(acc.values()) < 0.9999:
        if not scores_loaded:
            best_chan = None
            best_score = 0.0
        else:
            best_chan = max(scores_loaded, key=scores_loaded.get)
            best_score = scores_loaded[best_chan]
            scores_loaded.clear()

        # for every channel k in the E
        print(f"Search space: {search_space_}")
        for k in search_space_:
            keras.backend.clear_session()

            stats_on_K = 0
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
            stats_on_K = proposed_resnet18.train_model_on_V(
                V_tmp_data,
                V_tmp_labels,
                chosen_channels,
                saves_path,
                h,
                T,
                sliding_window_out,
                strategy,
            )
            print(f"Stats on channel(s) {chosen_channels} : {stats_on_K}")

            with open(acc_save_file, "a") as f:
                f.write(f"{chosen_channels} : {stats_on_K}\n")

            score = calculate_model_score_from_stats(stats_on_K)
            if score > best_score:
                best_score = score
                best_chan = k
            # scores[k] = calculate_model_score_from_stats(stats_on_K)
            tik_all[k] = tik
            print(f"Score on channel(s) {chosen_channels} : {score}")
            print("-------------------------------------------------")

        # values = np.array(list(scores.values()))
        # keys = np.array(list(scores.keys()))
        # k_star = int(keys[np.argmax(values)])

        # scores.clear()

        search_space_.discard(best_chan)
        best_channels.append(best_chan)
        print(f"Best channel(s) : {best_channels}")
        with open(acc_save_file, "a") as f:
            f.write(f"Best channel(s) : {best_channels}\n")
        print("==================================================")

        V = (
            tik_all[best_chan]
            if len(best_channels) == 1
            else np.concatenate([V, tik_all[best_chan]], axis=1)
        )
        tik_all.clear()
        i += 1


def channel_selection_resnet18_old(data, search_space, strategy, acc_save_file) -> None:
    # Detect already checked permutations from acc_save_file
    checked_permutations = set()
    print("Checking for already checked permutations in accuracy file...")
    if os.path.exists(acc_save_file):
        with open(acc_save_file, "r") as f:
            for line in f:
                # Each line format: "[channels] : stats"
                match = re.match(r"\[(.*?)\]\s*:", line)
                if match:
                    channels_str = match.group(1)
                    try:
                        channels = ast.literal_eval(f"[{channels_str}]")
                        checked_permutations.add(tuple(channels))
                    except Exception as e:
                        print(e)
                        continue
    print(f"Found {len(checked_permutations)} checked permutations")

    search_space_P = sorted(
        [
            p
            for p in permutations(search_space, 3)
            if tuple(p) not in checked_permutations
        ]
    )
    # symbol: i
    i = 0

    while True:
        print(f"Search space size: {len(search_space_P)}")
        # for every throuple of channels k in the search space
        for k in search_space_P:
            keras.backend.clear_session()
            V_tmp = []
            chosen_channels = list(k)

            V_tmp = orthogonalize_channels(data, k)
            V_tmp_data, V_tmp_labels = create_samples_and_labels(V_tmp, Gamma, D)

            print(f"Training on channels {chosen_channels}")
            stats_on_K = proposed_resnet18.train_model_on_V(
                V_tmp_data,
                V_tmp_labels,
                chosen_channels,
                saves_path,
                h,
                T,
                sliding_window_out,
                strategy,
            )
            # print(f"Stats on channels {chosen_channels}: {stats_on_K}")

            with open(acc_save_file, "a") as f:
                f.write(f"{chosen_channels} : {stats_on_K}\n")

            search_space_P.remove(k)
            print("==================================================")

        i += 1

        if len(search_space_P) == 0:
            break


def channel_selection_tcn(
    data, search_space, best_channels, strategy, acc_save_file
) -> None:
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

    keras.backend.clear_session()
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


def train_and_save_model(
    train_X, train_y, strategy, acc_savefile=None, model=None, best_channels=None, fname=None
) -> None:
    no_of_subjects = train_y.shape[1]

    # load best channels from acc_savefile
    if not best_channels:
        best_channels, _, _ = read_acc_file(acc_savefile)

    if acc_savefile:
        fname_prefix = acc_savefile.split("/")[-1].split(".")[0].split("_accuracies")[0]
    elif fname:
        fname_prefix = fname
    else:
        raise ValueError("Either acc_savefile or fname must be provided")

    X_train, X_test, y_train, y_test = train_test_split(
        train_X[:, best_channels, :], train_y, test_size=0.2, shuffle=True
    )

    dataset_train = create_dataset(
        X_train,
        y_train,
        sliding_window_out,
        strategy,
    )
    dataset_test = create_dataset(
        X_test,
        y_test,
        sliding_window_out,
        strategy,
    )

    with strategy.scope():
        if model == "cnn":
            callbacks = proposed_cnn.prepare_callbacks(saves_path, best_channels)
            model = proposed_cnn.create_model(
                no_of_subjects=no_of_subjects,
                h=h,
                T=T,
                no_of_channels=len(best_channels),
            )
        elif model == "resnet18":
            callbacks = proposed_resnet18.prepare_callbacks(saves_path, best_channels)
            model = proposed_resnet18.create_model(
                no_of_subjects=no_of_subjects,
                h=h,
                T=T,
                no_of_channels=len(best_channels),
            )
        else:
            raise ValueError("Model must be either 'cnn' or 'resnet18'")

        model.fit(
            x=dataset_train,
            validation_data=dataset_test,
            epochs=50,
            steps_per_epoch=None,
            validation_steps=None,
            verbose=1,
            callbacks=callbacks,
        )
    file_path = os.path.join(
        saves_path, f"{fname_prefix}_C_{best_channels}-final.keras"
    )
    model.save(file_path)
    print("==============================================")
    print("Job done")


def full_train_resnet18_model(data, strategy) -> None:
    # with strategy.scope():
    # data = orthogonalize_data(data)
    model = proposed_resnet18.create_model(
        no_of_channels=data.shape[1], no_of_subjects=data.shape[0], h=h, T=T
    )
    # model.summary()
    # print(data.shape)
    samples, labels = create_samples_and_labels(data, Gamma, D)
    # samples = tf.cast(samples, tf.float32)
    # print(samples.shape)
    train_X, test_X, train_y, test_y = train_test_split(
        samples.numpy(), labels.numpy(), test_size=0.2, shuffle=True
    )
    # print(train_X.shape)
    # dataset = create_dataset(samples, labels, sliding_window_out, strategy)
    dataset_train = create_dataset(train_X, train_y, sliding_window_out, strategy)
    # print([e for e in dataset_train.take(1)])
    dataset_test = create_dataset(test_X, test_y, sliding_window_out, strategy)

    before = datetime.now()
    stats = model.fit(
        x=dataset_train,
        validation_data=dataset_test,
        epochs=30,
        steps_per_epoch=None,
        validation_steps=None,
        verbose=1,
    )
    now = datetime.now()
    time_elapsed = str(now - before)
    results = {
        "time_elapsed": time_elapsed,
        "categorical_accuracy": stats.history["categorical_accuracy"][-1],
        "loss": stats.history["loss"][-1],
        "val_categorical_accuracy": stats.history["val_categorical_accuracy"][-1],
        "val_loss": stats.history["val_loss"][-1],
    }
    print(results)
    print("==============================================")
    print("Job done")
    # print(stats)


def calculate_and_save_thresholds(
    authenticator: EEGAuthenticator, test_X, test_y, fname
):
    filename = os.path.join(saves_path, f"{fname}_thresholds.txt")
    metrics_stats = []
    with open(filename, "a") as f:
        eucl_det = authenticator.calculate_threshold(
            distance.euclidean, test_X=test_X, test_y=test_y
        )
        eucl_threshold = eucl_det["optimal_threshold"]
        f.write(f"euclidean threshold: {eucl_threshold}\nEER: {eucl_det['eer']}\n\n")

        cos_det = authenticator.calculate_threshold(
            distance.cosine, test_X=test_X, test_y=test_y
        )
        cos_threshold = cos_det["optimal_threshold"]
        f.write(f"cosine threshold: {cos_threshold}\nEER: {cos_det['eer']}\n\n")

        manh_det = authenticator.calculate_threshold(
            distance.cityblock, test_X=test_X, test_y=test_y
        )
        manh_threshold = manh_det["optimal_threshold"]
        f.write(f"manhattan threshold: {manh_threshold}\nEER: {manh_det['eer']}\n\n")

    metrics_stats.append(eucl_det)
    metrics_stats.append(cos_det)
    metrics_stats.append(manh_det)

    # now plot all DET curves together
    fig, ax = plt.subplots()
    for stats in metrics_stats:
        fpr = stats["fpr"]
        fnr = stats["fnr"]
        eer = stats["eer"]
        eer_point = stats["eer_point"]

        ax.plot(fnr, fpr, label=f"{stats['distance_fn']}  ( {eer * 100:.3f} % )")
        ax.plot(fpr[eer_point], fnr[eer_point], "ro", zorder=10.1)  # mark the EER point

    ax.plot([0, 1], [0, 1], "k--", label="EER Line", scalex=False, scaley=False)
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(0.0, 0.6)

    ax.set_xlabel("False Rejection Rate")
    ax.set_ylabel("False Acceptance Rate")
    ax.set_title(f"{fname} DET Curves")
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(saves_path, f"{fname}_DET_curves.png"))

    print("Job done")
    print("==============================================")


def calculate_and_save_metrics(
    authenticator: EEGAuthenticator, test_X, test_y, fname, thresholds_file
):
    filename = os.path.join(saves_path, f"{fname}_metrics.txt")
    # thresholds_file = os.path.join(saves_path, f"{fname}_thresholds.txt")
    # read thresholds from file
    with open(thresholds_file, "r") as f:
        lines = f.readlines()
        eucl_threshold = float(lines[0].split(":")[1].strip())
        cos_threshold = float(lines[3].split(":")[1].strip())
        manh_threshold = float(lines[6].split(":")[1].strip())  
        # hardcoded because idgaf atp

    print(
        f"Loaded thresholds\neuclidean: {eucl_threshold}\ncosine: {cos_threshold}\nmanhattan: {manh_threshold}"
    )
    # metrics_stats = []
    with open(filename, "a") as f:
        eucl_stats = authenticator.calculate_metrics(
            distance.euclidean,
            test_X=test_X,
            test_y=test_y,
            threshold=eucl_threshold,
        )
        f.write(f"euclidean stats: {eucl_stats}\n")

        cos_stats = authenticator.calculate_metrics(
            distance.cosine,
            test_X=test_X,
            test_y=test_y,
            threshold=cos_threshold,
        )
        f.write(f"cosine stats: {cos_stats}\n")

        manh_stats = authenticator.calculate_metrics(
            distance.cityblock,
            test_X=test_X,
            test_y=test_y,
            threshold=manh_threshold,
        )
        f.write(f"manhattan stats: {manh_stats}\n")

    print("Job done")
    print("==============================================")


def calculate_and_save_confusion_matrices(
    authenticator: EEGAuthenticator, test_X, test_y, fname, thresholds_file
):
    # thresholds_file = os.path.join(saves_path, f"{fname}_thresholds.txt")
    # read thresholds from file
    with open(thresholds_file, "r") as f:
        lines = f.readlines()
        eucl_threshold = float(lines[0].split(":")[1].strip())
        cos_threshold = float(lines[3].split(":")[1].strip())
        manh_threshold = float(lines[6].split(":")[1].strip())  
        # hardcoded because idgaf atp

    print(
        f"Loaded thresholds\neuclidean: {eucl_threshold}\ncosine: {cos_threshold}\nmanhattan: {manh_threshold}"
    )
    authenticator.calculate_confusion_matrix(
        distance.euclidean,
        test_X=test_X,
        test_y=test_y,
        threshold=eucl_threshold,
        fname=fname,
        cmap="Blues",
    )
    authenticator.calculate_confusion_matrix(
        distance.cosine,
        test_X=test_X,
        test_y=test_y,
        threshold=cos_threshold,
        fname=fname,
        cmap="Oranges",
    )
    authenticator.calculate_confusion_matrix(
        distance.cityblock,
        test_X=test_X,
        test_y=test_y,
        threshold=manh_threshold,
        fname=fname,
        cmap="Greens",
    )
    
    print("Job done")
    print("==============================================")


def calculate_average_stopped_epoch(acc_save_file):
    stopped_epochs = []
    with open(acc_save_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "stopped_epoch" in line:
                m = line.strip().split("'stopped_epoch': ")[1].split(",")[0]
                stopped_epoch = int(m)
                stopped_epochs.append(stopped_epoch)
            else:
                continue
    print(acc_save_file)
    print(f"Average stopped epoch: {np.mean(stopped_epochs)}\n")


def get_all_test_user_samples(user_i: int, test_X, test_y):
    test_labels = np.argmax(test_y, axis=1)
    samples_to_keep = []
    for i, label in enumerate(test_labels):
        if label == user_i:
            samples_to_keep.append(i)
    return test_X[samples_to_keep, :, :]

def even_fewer_channels_tests():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    strategy = tf.distribute.get_strategy()

    all_20_channels = (
        np.array(
            [
                9,
                11,
                13,
                22,
                24,
                30,
                32,
                34,
                36,
                38,
                41,
                42,
                47,
                49,
                51,
                53,
                55,
                61,
                62,
                63,
            ]
        )
        - 1
    )
    all_20_channels = list(int(c) for c in all_20_channels)
    important_channels = np.array([22, 23, 24, 25, 26, 27, 28, 29]) - 1
    important_channels = list(int(c) for c in important_channels)
    
    chosen_channels = set(all_20_channels)
    chosen_channels = chosen_channels.union(important_channels)
    chosen_channels = sorted(list(chosen_channels))
    
    # we need to map the best_channels to the indices of the chosen_channels array
    best_channels1 = [chosen_channels.index(c) for c in [40]]  
    best_channels2 = [chosen_channels.index(c) for c in [40, 41]]
    best_channels3 = [chosen_channels.index(c) for c in [28]]
    best_channels4 = [chosen_channels.index(c) for c in [28, 21]]

    alpha_train_X = np.load("user_data/alpha_train_X.npy")
    alpha_train_y = np.load("user_data/alpha_train_y.npy")
    
    # train_and_save_model(
    #     train_X=alpha_train_X, train_y=alpha_train_y, strategy=strategy, best_channels=best_channels1, model="resnet18", fname="all20_resnet18"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     train_X=alpha_train_X, train_y=alpha_train_y, strategy=strategy, best_channels=best_channels2, model="resnet18", fname="all20_resnet18"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     train_X=alpha_train_X, train_y=alpha_train_y, strategy=strategy, best_channels=best_channels3, model="resnet18", fname="frontal_resnet18"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     train_X=alpha_train_X, train_y=alpha_train_y, strategy=strategy, best_channels=best_channels4, model="resnet18", fname="frontal_resnet18"
    # )
    
    model1 = "saves2/all20_resnet18_C_[16]-final.keras"
    model2 = "saves2/all20_resnet18_C_[16, 17]-final.keras"
    model3 = "saves2/frontal_resnet18_C_[10]-final.keras"
    model4 = "saves2/frontal_resnet18_C_[10, 3]-final.keras"
    
    authenticator1 = EEGAuthenticator(
        user_data=alpha_train_X[:, best_channels1, :],
        user_labels=alpha_train_y,
        model=model1,
        Gamma=Gamma,
        D=D,
        T=T,
        delta=delta,
        strategy=strategy,
        random_state=42,
    )
    authenticator2 = EEGAuthenticator(
        user_data=alpha_train_X[:, best_channels2, :],
        user_labels=alpha_train_y,
        model=model2,
        Gamma=Gamma,
        D=D,
        T=T,
        delta=delta,
        strategy=strategy,
        random_state=42,
    )
    authenticator3 = EEGAuthenticator(
        user_data=alpha_train_X[:, best_channels3, :],
        user_labels=alpha_train_y,
        model=model3,
        Gamma=Gamma,
        D=D,
        T=T,
        delta=delta,
        strategy=strategy,
        random_state=42,
    )
    authenticator4 = EEGAuthenticator(
        user_data=alpha_train_X[:, best_channels4, :],
        user_labels=alpha_train_y,
        model=model4,
        Gamma=Gamma,
        D=D,
        T=T,
        delta=delta,
        strategy=strategy,
        random_state=42,
    )
    
    alpha_test_X = np.load("user_data/alpha_test_X.npy")
    alpha_test_y = np.load("user_data/alpha_test_y.npy")
    
    # calculate_and_save_thresholds(authenticator1, alpha_test_X[:, best_channels1, :], alpha_test_y, "all20_resnet18_few")
    # keras.backend.clear_session()
    # calculate_and_save_thresholds(authenticator2, alpha_test_X[:, best_channels2, :], alpha_test_y, "all20_resnet18_fewer")
    # keras.backend.clear_session()
    calculate_and_save_thresholds(authenticator3, alpha_test_X[:, best_channels3, :], alpha_test_y, "frontal_resnet18_few")
    keras.backend.clear_session()
    # calculate_and_save_thresholds(authenticator4, alpha_test_X[:, best_channels4, :], alpha_test_y, "frontal_resnet18_fewer")
    # keras.backend.clear_session()
    

def main() -> None:
    even_fewer_channels_tests()
    return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    strategy = tf.distribute.get_strategy()
    # data = load_data(dataset_path=dataset_path, end=108)

    all_20_channels = (
        np.array(
            [
                9,
                11,
                13,
                22,
                24,
                30,
                32,
                34,
                36,
                38,
                41,
                42,
                47,
                49,
                51,
                53,
                55,
                61,
                62,
                63,
            ]
        )
        - 1
    )
    all_20_channels = list(int(c) for c in all_20_channels)
    important_channels = np.array([22, 23, 24, 25, 26, 27, 28, 29]) - 1
    important_channels = list(int(c) for c in important_channels)
    
    bidgoly_channels = np.array([62, 41, 11]) - 1
    # full_train_resnet18_model(data=data, strategy=strategy)
    # channel_selection_resnet18(
    #     data=data,
    #     search_space=all_20_channels,
    #     acc_save_file=os.path.join(saves_path, "all20_resnet18_accuracies.txt"),
    #     strategy=strategy,
    # )
    # channel_selection_CNN(
    #     data=data,
    #     search_space=important_channels,
    #     acc_save_file=os.path.join(saves_path, "frontal_cnn_accuracies.txt"),
    #     strategy=strategy,
    # )
    # channel_selection_resnet18(
    #     data=data,
    #     search_space=important_channels,
    #     acc_save_file=os.path.join(saves_path, "frontal_resnet18_accuracies.txt"),
    #     strategy=strategy,
    # )

    # read_all_scores(acc_save_file=os.path.join(saves_path, "resnet18_accuracies.txt"))
    # calculate_resnet18_model_score("saves2/accuracies.txt")
    # chosen_channels = [22, 23, 24]

    # print(f"Data shape: {beta_users.shape}")
    chosen_channels = set(all_20_channels)
    chosen_channels = chosen_channels.union(important_channels)
    chosen_channels = sorted(list(chosen_channels))

    # prepare_and_save_user_data(
    #     alpha_users_n=90, alpha_test_size=0.2, chosen_channels=chosen_channels, strategy=strategy
    # )

    # acc_save_file1 = "saves2/all20_resnet18_accuracies.txt"
    # thresholds_file1 = "saves2/all20_resnet18_thresholds.txt"
    # best_channels1, _, _ = read_acc_file(acc_save_file1)
    # best_channels1 = [
    #     chosen_channels.index(c) for c in best_channels1
    # ]  # we need to map the best_channels to the indices of the chosen_channels array

    # acc_save_file2 = "saves2/frontal_cnn_accuracies.txt"
    # thresholds_file2 = "saves2/frontal_cnn_thresholds.txt"
    # best_channels2, _, _ = read_acc_file(acc_save_file2)
    # best_channels2 = [chosen_channels.index(c) for c in best_channels2]

    # acc_save_file3 = "saves2/frontal_resnet18_accuracies.txt"
    # thresholds_file3 = "saves2/frontal_resnet18_thresholds.txt"
    # best_channels3, _, _ = read_acc_file(acc_save_file3)
    # best_channels3 = [chosen_channels.index(c) for c in best_channels3]
    
    # acc_save_file4 = "saves2/all20_cnn_accuracies.txt"
    thresholds_file4 = "saves2/all20_cnn_thresholds.txt"
    best_channels4 = [chosen_channels.index(c) for c in bidgoly_channels]

    alpha_train_X = np.load("user_data/alpha_train_X.npy")
    alpha_train_y = np.load("user_data/alpha_train_y.npy")

    # calculate_average_stopped_epoch(acc_save_file1)
    # calculate_average_stopped_epoch(acc_save_file2)
    # calculate_average_stopped_epoch(acc_save_file3)
    
    # train_and_save_model(
    #     alpha_train_X, alpha_train_y, acc_save_file1, strategy, best_channels=best_channels1, model="resnet18"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     alpha_train_X, alpha_train_y, acc_save_file2, strategy, best_channels=best_channels2, model="cnn"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     alpha_train_X, alpha_train_y, acc_save_file3, strategy, best_channels=best_channels3, model="resnet18"
    # )
    # keras.backend.clear_session()
    # train_and_save_model(
    #     alpha_train_X, alpha_train_y, acc_save_file4, strategy, best_channels=best_channels4, model="cnn"
    # )

    # print(f"{alpha_test_X.shape}, {alpha_test_y.shape}")

    # model1 = "saves2/all20_resnet18_C_[16, 17, 1]-final.keras"
    # model2 = "saves2/frontal_cnn_C_[7, 3, 10]-final.keras"
    # model3 = "saves2/frontal_resnet18_C_[10, 3, 9]-final.keras"
    model4 = "saves2/all20_cnn_C_[24, 16, 1]-final.keras"

    ## alpha tests
    alpha_test_X = np.load("user_data/alpha_test_X.npy")
    alpha_test_y = np.load("user_data/alpha_test_y.npy")
    #
    # authenticator1 = EEGAuthenticator(
    #     user_data=alpha_train_X[:, best_channels1, :],
    #     user_labels=alpha_train_y,
    #     model=model1,
    #     Gamma=Gamma,
    #     D=D,
    #     T=T,
    #     delta=delta,
    #     strategy=strategy,
    #     random_state=42,
    # )
    # authenticator2 = EEGAuthenticator(
    #     user_data=alpha_train_X[:, best_channels2, :],
    #     user_labels=alpha_train_y,
    #     model=model2,
    #     Gamma=Gamma,
    #     D=D,
    #     T=T,
    #     delta=delta,
    #     strategy=strategy,
    #     random_state=42,
    # )
    # authenticator3 = EEGAuthenticator(
    #     user_data=alpha_train_X[:, best_channels3, :],
    #     user_labels=alpha_train_y,
    #     model=model3,
    #     Gamma=Gamma,
    #     D=D,
    #     T=T,
    #     delta=delta,
    #     strategy=strategy,
    #     random_state=42,
    # )
    authenticator4 = EEGAuthenticator(
        user_data=alpha_train_X[:, best_channels4, :],
        user_labels=alpha_train_y,
        model=model4,
        Gamma=Gamma,
        D=D,
        T=T,
        delta=delta,
        strategy=strategy,
        random_state=42,
    )

    # calculate_and_save_thresholds(authenticator1, alpha_test_X[:, best_channels1, :], alpha_test_y, "all20_resnet18")
    # calculate_and_save_thresholds(authenticator2, alpha_test_X[:, best_channels2, :], alpha_test_y, "frontal_cnn")
    # calculate_and_save_thresholds(authenticator3, alpha_test_X[:, best_channels3, :], alpha_test_y, "frontal_resnet18")
    # calculate_and_save_thresholds(authenticator4, alpha_test_X[:, best_channels4, :], alpha_test_y, "all20_cnn")

    # calculate_and_save_metrics(
    #     authenticator1,
    #     alpha_test_X[:, best_channels1, :],
    #     alpha_test_y,
    #     "all20_resnet18_alpha",
    # )
    # calculate_and_save_metrics(
    #     authenticator2,
    #     alpha_test_X[:, best_channels2, :],
    #     alpha_test_y,
    #     "frontal_cnn_alpha",
    # )
    # calculate_and_save_metrics(
    #     authenticator3,
    #     alpha_test_X[:, best_channels3, :],
    #     alpha_test_y,
    #     "frontal_resnet18_alpha",
    # )
    # calculate_and_save_metrics(
    #     authenticator4,
    #     alpha_test_X[:, best_channels4, :],
    #     alpha_test_y,
    #     "all20_cnn_alpha",
    #     thresholds_file4,
    # )
    
    # calculate_and_save_confusion_matrices(
    #     authenticator1,
    #     alpha_test_X[:, best_channels1, :],
    #     alpha_test_y,
    #     "all20_resnet18_alpha",
    # )
    # keras.backend.clear_session()
    # calculate_and_save_confusion_matrices(
    #     authenticator2,
    #     alpha_test_X[:, best_channels2, :],
    #     alpha_test_y,
    #     "frontal_cnn_alpha",
    # )
    # keras.backend.clear_session()
    # calculate_and_save_confusion_matrices(
    #     authenticator3,
    #     alpha_test_X[:, best_channels3, :],
    #     alpha_test_y,
    #     "frontal_resnet18_alpha",
    # )
    # keras.backend.clear_session()
    # calculate_and_save_confusion_matrices(
    #     authenticator4,
    #     alpha_test_X[:, best_channels4, :],
    #     alpha_test_y,
    #     "all20_cnn_alpha",
    #     thresholds_file4,
    # )
    # ## end alpha tests
    
    # ## beta tests
    # betas_X = np.load("user_data/beta_X.npy")
    # betas_y = np.load("user_data/beta_y.npy")
    
    # beta_train_X, beta_test_X, beta_train_y, beta_test_y = train_test_split(
    #     betas_X, betas_y, test_size=0.2, shuffle=True
    # )

    # # beta_authenticator1 = EEGAuthenticator(
    # #     user_data=beta_train_X[:, best_channels1, :],
    # #     user_labels=beta_train_y,
    # #     model=model1,
    # #     Gamma=Gamma,
    # #     D=D,
    # #     T=T,
    # #     delta=delta,
    # #     strategy=strategy,
    # #     random_state=42,
    # # )
    # # beta_authenticator2 = EEGAuthenticator(
    # #     user_data=beta_train_X[:, best_channels2, :],
    # #     user_labels=beta_train_y,
    # #     model=model2,
    # #     Gamma=Gamma,
    # #     D=D,
    # #     T=T,
    # #     delta=delta,
    # #     strategy=strategy,
    # #     random_state=42,
    # # )
    # # beta_authenticator3 = EEGAuthenticator(
    # #     user_data=beta_train_X[:, best_channels3, :],
    # #     user_labels=beta_train_y,
    # #     model=model3,
    # #     Gamma=Gamma,
    # #     D=D,
    # #     T=T,
    # #     delta=delta,
    # #     strategy=strategy,
    # #     random_state=42,
    # # )
    # beta_authenticator4 = EEGAuthenticator(
    #     user_data=beta_train_X[:, best_channels4, :],
    #     user_labels=beta_train_y,
    #     model=model4,
    #     Gamma=Gamma,
    #     D=D,
    #     T=T,
    #     delta=delta,
    #     strategy=strategy,
    #     random_state=42,
    # )
    # # calculate_and_save_metrics(
    # #     beta_authenticator1,
    # #     beta_test_X[:, best_channels1, :],
    # #     beta_test_y,
    # #     "all20_resnet18_beta",
    # #     thresholds_file1,
    # # )
    # # calculate_and_save_metrics(
    # #     beta_authenticator2,
    # #     beta_test_X[:, best_channels2, :],
    # #     beta_test_y,
    # #     "frontal_cnn_beta",
    # #     thresholds_file2
    # # )
    # # calculate_and_save_metrics(
    # #     beta_authenticator3,
    # #     beta_test_X[:, best_channels3, :],
    # #     beta_test_y,
    # #     "frontal_resnet18_beta",
    # #     thresholds_file3
    # # )
    # calculate_and_save_metrics(
    #     beta_authenticator4,
    #     beta_test_X[:, best_channels4, :],
    #     beta_test_y,
    #     "all20_cnn_beta",
    #     thresholds_file4
    # )
    
    # # calculate_and_save_confusion_matrices(
    # #     beta_authenticator1,
    # #     beta_test_X[:, best_channels1, :],
    # #     beta_test_y,
    # #     "all20_resnet18_beta",
    # #     thresholds_file1,
    # # )
    # # keras.backend.clear_session()
    # # calculate_and_save_confusion_matrices(
    # #     beta_authenticator2,
    # #     beta_test_X[:, best_channels2, :],
    # #     beta_test_y,
    # #     "frontal_cnn_beta",
    # #     thresholds_file2,
    # # )
    # # keras.backend.clear_session()
    # # calculate_and_save_confusion_matrices(
    # #     beta_authenticator3,
    # #     beta_test_X[:, best_channels3, :],
    # #     beta_test_y,
    # #     "frontal_resnet18_beta",
    # #     thresholds_file3,
    # # )
    # keras.backend.clear_session()
    # calculate_and_save_confusion_matrices(
    #     beta_authenticator4,
    #     beta_test_X[:, best_channels4, :],
    #     beta_test_y,
    #     "all20_cnn_beta",
    #     thresholds_file4,
    # )
    ## end beta tests


if __name__ == "__main__":
    main()
