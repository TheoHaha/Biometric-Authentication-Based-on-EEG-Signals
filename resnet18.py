import os

import tensorflow as tf
from classification_models.keras import Classifiers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (  # type:ignore  # noqa: F401
    ModelCheckpoint,
)
from tensorflow.keras.losses import CategoricalCrossentropy  # type:ignore
from tensorflow.keras.optimizers import RMSprop  # type: ignore

from my_utils import checkpoint_exists, create_dataset, get_latest_epoch  # type:ignore

ResNet18, preprocess_input = Classifiers.get("resnet18")


def create_model(no_of_subjects=90, h=20, T=160, trained=True):
    weights = "imagenet" if trained else ""
    base_model = ResNet18(
        input_shape=(h, T, 3),
        weights=weights,
        classes=no_of_subjects,
        include_top=False,
    )

    fingerprint_layers = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.Flatten(),
        ]
    )

    id_layers = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dropout(rate=0.25),
            tf.keras.layers.Dense(no_of_subjects, activation="softmax"),
        ]
    )

    model = tf.keras.Sequential(
        [
            fingerprint_layers,
            id_layers,
        ]
    )

    optimizer = RMSprop(learning_rate=0.0001)  # , clipvalue=1.0, centered=False)
    loss = CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            # tf.keras.metrics.Precision(),
            # tf.keras.metrics.Recall(),
        ],
    )

    return model


# custom callback that deletes saved model of 2nd previous epoch
class DeleteCheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_file_dir, chosen_channels, max_epochs):
        self.save_file_dir = save_file_dir
        self.chosen_channels = chosen_channels
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            checkpoint_path = os.path.join(
                self.save_file_dir,
                f"resnet18_C_{self.chosen_channels}-epoch_{epoch - 1:02d}.keras",
            )
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


def train_model_on_V(
    V, labels, chosen_channels, save_file_dir, h, T, sliding_window_out, strategy
):
    # no_of_channels = V.shape[1]
    no_of_subjects = labels.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(
        V.numpy(), labels.numpy(), test_size=0.2, random_state=42
    )

    with strategy.scope():
        dataset_train = create_dataset(X_train, y_train, sliding_window_out, strategy)
        dataset_test = create_dataset(X_test, y_test, sliding_window_out, strategy)
        # create save file directory if it doesn't exist
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
            print(f"Created directory {save_file_dir}")

        # set up checkpoint callback with the given save file directory
        cp_file_name = f"resnet18_C_{chosen_channels}" + "-epoch_{epoch:02d}.keras"
        checkpoint_path = os.path.join(save_file_dir, cp_file_name)
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=False, verbose=1
        )
        del_checkpoint_callback = DeleteCheckpointCallback(
            save_file_dir, chosen_channels, max_epochs=30
        )
        # checkpoint_callback = BackupAndRestore(
        #     backup_dir=checkpoint_path, save_freq="epoch", delete_checkpoint=True
        # )

        if not checkpoint_exists(save_file_dir, chosen_channels):
            # if no existing checkpoints are found, create the model from scratch and set initial epoch to 0
            print("no existing checkpoints found")
            model = create_model(no_of_subjects=no_of_subjects)
            latest_epoch = 0
        else:
            # if we find checkpoints, pick up where we left off
            latest_epoch, model_dir = get_latest_epoch(save_file_dir, chosen_channels)
            model = tf.keras.models.load_model(model_dir)
            print(f"loaded model from {model_dir}")
            print(f"resuming training from epoch {latest_epoch + 1}")

        stats = model.fit(
            x=dataset_train,
            validation_data=dataset_test,
            epochs=30,
            initial_epoch=latest_epoch,
            steps_per_epoch=None,
            validation_steps=None,
            verbose=1,
            callbacks=[checkpoint_callback, del_checkpoint_callback],
        )
    # pdb.set_trace()
    return stats.history["categorical_accuracy"][-1]


# ResNet18(input_shape=(20,160,3), weights='imagenet', include_top=False).summary()
# create_model().summary()
