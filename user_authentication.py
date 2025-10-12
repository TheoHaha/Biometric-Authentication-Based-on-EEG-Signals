import random
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from alive_progress import alive_bar
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    det_curve,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

from my_utils import sliding_window


def get_fingerprinting_from_model(model):
    if isinstance(model, str):
        model = keras.models.load_model(model)
    elif isinstance(model, keras.Model):
        pass
    else:
        raise ValueError("model must be a file path or a keras.Model instance")

    fingerprinting_model = model.layers[0]
    return fingerprinting_model


class EEGAuthenticator:
    def __init__(
        self,
        user_data,
        user_labels,
        model: str | keras.Model,
        Gamma: int,
        D: int,
        T: int,
        delta: int,
        strategy: tf.distribute.Strategy,
        random_state: int | None = None,
    ):
        self.user_data = user_data
        self.user_labels = user_labels
        self.fingerprint_fn = get_fingerprinting_from_model(model)
        self.__Gamma = Gamma
        self.__D = D
        self.__T = T
        self.__delta = delta
        self.__strategy = strategy
        self.__rs = np.random.RandomState(random_state)

    def authenticate_user(
        self, A: int, S, distance_fn, threshold=0.0
    ) -> tuple[float, bool]:
        fp_a = self.get_fingerprint_of_user_i(A)

        if isinstance(S, int):
            fp_s = self.get_fingerprint_of_user_i(S)
        else:
            fp_s = self.get_fingerprint_from_data(S)

        dist = distance_fn(fp_a, fp_s)
        return dist, dist <= threshold

    def get_fingerprint_of_user_i(self, i: int, n=1) -> np.ndarray:
        a = self.__get_all_user_samples(i)
        a = random.sample(list(a), n)  # use n random samples to get fingerprint
        a = np.array(a)
        a = self.__get_data_ready_for_fingerprinting(a)
        fp_a = self.fingerprint_fn(a)
        return fp_a

    def get_fingerprint_from_data(self, data: np.ndarray, n=1) -> np.ndarray:
        a = random.sample(list(data), n)  # use n random samples to get fingerprint
        a = np.array(a)
        a = self.__get_data_ready_for_fingerprinting(a)
        fp_a = self.fingerprint_fn(a)
        return fp_a

    def __get_all_user_samples(self, user_i: int):
        labels = np.argmax(self.user_labels, axis=1)
        samples_to_keep = []
        for i, label in enumerate(labels):
            if label == user_i:
                samples_to_keep.append(i)

        return self.user_data[samples_to_keep, :, :]

    # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html
    def calculate_threshold(self, distance_fn, test_X, test_y, plot=False) -> float:
        # user_count = 90
        fp_count = 1

        test_fingerprints, test_labels_y = self.__get_user_fingerprints_from_data(
            test_X, test_y, fp_count
        )
        user_data_fingerprints, user_labels = (
            self.__get_user_fingerprints_from_user_data(fp_count)
        )

        distances = cdist(
            user_data_fingerprints, test_fingerprints, metric=distance_fn.__name__
        )
        distances = distances.flatten()
        # print(f"distances shape: {distances.shape}")
        # return 0.0
        y_true = []
        for user_label_i in user_labels:
            for test_label_i in test_labels_y:
                if int(user_label_i) == int(test_label_i):
                    y_true.append(1)
                else:
                    y_true.append(0)
        y_true = np.array(y_true)
        # print(f"y_true shape: {y_true.shape}")
        # print(f"y_true max: {max(y_true)}")
        # print(f"y_true min: {min(y_true)}")

        # Calculate DET curve
        fpr, fnr, thresholds = det_curve(y_true, -distances)

        # Find EER point (where FPR =~ FNR)
        eer_point = np.nanargmin(np.absolute(fpr - fnr))
        eer = np.mean([fpr[eer_point], fnr[eer_point]])

        # Convert threshold back to original distance scale
        optimal_threshold = -thresholds[eer_point]

        # Plot DET curve
        if plot:
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, fnr)
            plt.plot([0, 1], [0, 1], "k--")  # diagonal line
            plt.plot(fpr[eer_point], fnr[eer_point], "ro", label=f"EER = {eer:.3f}")
            plt.xlim = [0.0, 0.6]
            plt.ylim = [0.0, 0.6]

            plt.xlabel("False Positive Rate")
            plt.ylabel("False Negative Rate")
            plt.title(f"DET Curve ({distance_fn.__name__} distance)")
            plt.suptitle(
                f"Optimal threshold {optimal_threshold:.4f}, EER: {eer * 100:.3f}%"
            )
            plt.legend()
            plt.grid(True)
            plt.show()

        # print(f"Optimal threshold at EER: {optimal_threshold:.4f}")
        result = {
            "optimal_threshold": optimal_threshold,
            "fpr": fpr,
            "fnr": fnr,
            "eer": eer,
            "eer_point": eer_point,
            "distance_fn": distance_fn.__name__,
        }
        return result

    def calculate_metrics(self, distance_fn, test_X, test_y, threshold):
        # user_count = 90
        fp_count = 10

        test_fingerprints, test_labels_y = self.__get_user_fingerprints_from_data(
            test_X, test_y, fp_count
        )
        user_fingerprints, user_labels_y = self.__get_user_fingerprints_from_user_data(
            fp_count
        )

        distances = cdist(
            user_fingerprints, test_fingerprints, metric=distance_fn.__name__
        )
        distances = distances.flatten()

        y_true = np.where(np.array([user_labels_y]).T == test_labels_y, 1, 0).flatten()

        # prepare y_pred according to threshold
        y_pred = (distances <= threshold).astype(int)

        accuracy = accuracy_score(y_true, y_pred)
        precicion = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precicion,
            "recall": recall,
        }

    def calculate_confusion_matrix(self, distance_fn, test_X, test_y, threshold, fname, cmap='Blues'):
        # user_count = 90
        fp_count = 10

        test_fingerprints, test_labels_y = self.__get_user_fingerprints_from_data(
            test_X, test_y, fp_count
        )
        user_fingerprints, user_labels_y = self.__get_user_fingerprints_from_user_data(
            fp_count
        )

        distances = cdist(
            user_fingerprints, test_fingerprints, metric=distance_fn.__name__
        )
        distances = distances.flatten()

        y_true = np.where(np.array([user_labels_y]).T == test_labels_y, 1, 0).flatten()

        # prepare y_pred according to threshold
        y_pred = (distances <= threshold).astype(int)
        
        disp1 = ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["Impostor", "Genuine"],
            cmap=cmap,
            normalize="true",
        )
        disp1.ax_.set_title(f"{fname} Confusion Matrix for {distance_fn.__name__} distance")
        disp1.figure_.savefig(f"{fname}_confusion_matrix_{distance_fn.__name__}.png")
    
    def __get_user_fingerprints_from_data(self, data_X, data_y, fp_count=1):
        data_labels = np.argmax(data_y, axis=1)
        unique_data_users = np.unique(data_labels)  # [:user_count]
        data_fingerprints = []
        data_labels_y = []
        for i in unique_data_users:
            data_samples_i = data_X[data_labels == i]
            data_fingerprints_i = self.get_fingerprint_from_data(
                data_samples_i, fp_count
            )
            data_fingerprints.append(data_fingerprints_i)

            data_labels_i = [i] * len(data_fingerprints_i)
            data_labels_y.append(data_labels_i)
        data_fingerprints = np.concatenate(data_fingerprints, axis=0)
        data_labels_y = np.concatenate(data_labels_y, axis=0)

        return data_fingerprints, data_labels_y

    def __get_user_fingerprints_from_user_data(self, fp_count=1):
        user_data_fingerprints = []
        user_labels = []
        unique_users = np.unique(np.argmax(self.user_labels, axis=1))  # [:user_count]
        for i in unique_users:
            user_fingerprint = self.get_fingerprint_of_user_i(i, fp_count)
            user_data_fingerprints.append(user_fingerprint)
            user_labels_i = [i] * len(user_fingerprint)
            user_labels.append(user_labels_i)
        user_data_fingerprints = np.concatenate(user_data_fingerprints, axis=0)
        user_labels = np.concatenate(user_labels, axis=0)

        return user_data_fingerprints, user_labels

    def __get_data_ready_for_fingerprinting(self, data) -> np.ndarray:
        with self.__strategy.scope():
            # test_data = sliding_window(data, self.__Gamma, self.__D)
            test_data = sliding_window(data, self.__T, self.__delta)
            # print(test_data.shape)

        if len(test_data.shape) == 3:
            test_data = np.expand_dims(test_data, axis=0)

        test_data = tf.transpose(test_data, perm=[0, 2, 3, 1])

        return test_data
