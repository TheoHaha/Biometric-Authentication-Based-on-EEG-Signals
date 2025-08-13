import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from alive_progress import alive_bar
from scipy.spatial.distance import cdist

from my_utils import sliding_window


def get_fingerprinting_from_model(model):
    fingerprinting_model = model.layers[0]
    return fingerprinting_model


class EEGAuthenticator:
    def __init__(
        self,
        user_data,
        user_labels,
        model,
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

    def get_fingerprint_of_user_i(self, i: int) -> np.ndarray:
        # a = self.__get_data_ready_for_fingerprinting(A)
        a = self.__get_all_user_samples(i)
        a = self.__get_data_ready_for_fingerprinting(a)
        fp_a = np.mean(self.fingerprint_fn(a), axis=0)
        return fp_a

    def get_fingerprint_from_data(self, data: np.ndarray) -> np.ndarray:
        s = self.__get_data_ready_for_fingerprinting(data)
        fp_s = np.mean(self.fingerprint_fn(s), axis=0)
        return fp_s

    def calculate_cdist(self, A, S, distance_fn):
        if isinstance(A, int):
            a = self.__get_all_user_samples(A)
            a = self.__get_data_ready_for_fingerprinting(a)
        else:
            a = self.__get_data_ready_for_fingerprinting(A)

        if isinstance(S, int):
            s = self.__get_all_user_samples(S)
            s = self.__get_data_ready_for_fingerprinting(s)
        else:
            s = self.__get_data_ready_for_fingerprinting(S)

        # min_samples = min(len(a), len(s))
        # a = np.array(a)[self.__rs.choice(len(a), min_samples, replace=False),:,:]
        # s = np.array(s)[self.__rs.choice(len(s), min_samples, replace=False),:,:]

        # fp_a = self.fingerprint_fn(a)
        # fp_s = self.fingerprint_fn(s)

        fp_a = []
        for i, _ in enumerate(a):
            f = np.array(self.fingerprint_fn(a[i : i + 1, :, :])).flatten()
            fp_a.append(f)
        # fp_a = np.array(fp_a)

        fp_s = []
        for i, _ in enumerate(s):
            f = np.array(self.fingerprint_fn(s[i : i + 1, :, :])).flatten()
            fp_s.append(f)
        # fp_s = np.array(fp_s)

        min_samples = min(len(fp_a), len(fp_s))
        fp_a = np.array(fp_a)[
            self.__rs.choice(len(fp_a), min_samples, replace=False), :
        ]
        fp_s = np.array(fp_s)[
            self.__rs.choice(len(fp_s), min_samples, replace=False), :
        ]

        print(f"fp_a shape: {fp_a.shape}")
        print(f"fp_s shape: {fp_s.shape}")

        dist = cdist(fp_a, fp_s, metric=distance_fn)
        print(dist.shape)
        return 0.1

    def __get_all_user_samples(self, user_i: int):
        labels = np.argmax(self.user_labels, axis=1)
        samples_to_keep = []
        for i, label in enumerate(labels):
            if label == user_i:
                samples_to_keep.append(i)

        return self.user_data[samples_to_keep, :, :]

    # TODO: replace with DET curve calculations
    # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_det.html
    def calculate_threshold(self, distance_fn, test_X, test_y) -> float:
        test_labels = np.argmax(test_y, axis=1)

        unique_users = np.unique(np.argmax(self.user_labels, axis=1))[:10]
        unique_test_users = np.unique(test_labels)[:10]

        bar_total = len(unique_users) * len(unique_test_users)

        def get_all_test_user_samples(user_i: int):
            samples_to_keep = []
            for i, label in enumerate(test_labels):
                if label == user_i:
                    samples_to_keep.append(i)
            return test_X[samples_to_keep, :, :]

        # construct the y_true tables for each user
        y_true = []
        with alive_bar(
            total=bar_total,
            title="Constructing y_true table",
        ) as bar:
            for user_i in unique_users:
                # y_true_i = []
                for test_user_i in unique_test_users:
                    if test_user_i == user_i:
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    bar()
                # y_trues.append(y_true_i)

        y_true = np.array(y_true)
        # print(f"y_trues shape: {y_trues.shape}")

        # construct the y_scores tables for each user
        # clf = SVC(random_state=self.__rs).fit(self.user_data, self.user_labels)
        # y_scores = clf.decision_function(test_X)
        y_scores = []
        with self.__strategy.scope():
            with alive_bar(
                total=bar_total,
                title="Constructing y_scores tables",
            ) as bar:
                for user_i in unique_users:
                    for user_j in unique_test_users:
                        user_j_samples = get_all_test_user_samples(user_j)
                        y_score, _ = self.authenticate_user(
                            user_i, user_j_samples, distance_fn
                        )

                        y_scores.append(y_score)
                        bar()

        y_scores = np.array(y_scores)

        min_score = min(y_scores)
        max_score = max(y_scores)
        no_of_thresholds = 25
        thresholds = np.linspace(min_score, max_score, no_of_thresholds)
        print(thresholds)

        # constructing y_pred table for each threshold
        y_preds = []
        with alive_bar(
            total=no_of_thresholds * bar_total,
            title="Constructing y_pred for each threshold",
        ) as bar:
            for threshold in thresholds:
                y_pred_i = []
                for dist in y_scores:
                    if dist < threshold:
                        y_pred_i.append(1)
                    else:
                        y_pred_i.append(0)
                    bar()
                y_preds.append(y_pred_i)
        y_preds = np.array(y_preds)
        
        fpr = []
        fnr = []
        with alive_bar(total=no_of_thresholds, title="Calculating fpr and fnr") as bar:
            for y_pred in y_preds:
                # print(f"y_pred={y_pred}")
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                print(f"tp={tp}, tn={tn}, fp={fp}, fn={fn}")
                
                fpr_tmp = fp / (fp + tn) if (fp + tn) > 0 else 0
                fpr.append(float(fpr_tmp))
                
                fnr_tmp = fn / (tp + fn) if (tp + fn) > 0 else 0
                fnr.append(float(fnr_tmp))
                
                bar()
        
        print(f"FPR={fpr}")
        print(f"FNR={fnr}")
        
        fig, ax = plt.subplots()
        ax.plot(fpr, fnr, 'ro')
        ax.xaxis.set_label_text("FPR")
        ax.yaxis.set_label_text("FNR")
        plt.show()
        
        return 1.0

    def confusion_matrix(self, distance_fn, test_X, test_y, threshold):
        test_labels = np.argmax(test_y, axis=1)

        unique_users = np.unique(np.argmax(self.user_labels, axis=1))
        unique_test_users = np.unique(test_labels)

        def get_all_test_user_samples(user_i: int):
            samples_to_keep = []
            for i, label in enumerate(test_labels):
                if label == user_i:
                    samples_to_keep.append(i)
            return test_X[samples_to_keep, :, :]

        bar_total = len(unique_users) * len(unique_test_users)

        # construct the y_true tables for each user
        y_true = []
        with alive_bar(total=bar_total, title="Constructing y_true tables") as bar:
            for user_i in unique_users:
                for user_j in unique_test_users:
                    if user_j == user_i:
                        y_true.append(1)
                    else:
                        y_true.append(0)
                    bar()

        y_true = np.array(y_true)

        # construct the y_pred tables
        y_pred = []
        with alive_bar(total=bar_total, title="Constructing y_pred table") as bar:
            for user_i in unique_users:
                for user_j in unique_test_users:
                    user_j_samples = get_all_test_user_samples(user_j)
                    _, authed = self.authenticate_user(
                        user_i, user_j_samples, distance_fn, threshold
                    )
                    if authed:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                    bar()

        y_pred = np.array(y_pred)

    def __get_data_ready_for_fingerprinting(self, data) -> np.ndarray:
        with self.__strategy.scope():
            # test_data = sliding_window(data, self.__Gamma, self.__D)
            test_data = sliding_window(data, self.__T, self.__delta)
            # print(test_data.shape)

        test_data = tf.transpose(test_data, perm=[0, 2, 3, 1])

        return test_data
