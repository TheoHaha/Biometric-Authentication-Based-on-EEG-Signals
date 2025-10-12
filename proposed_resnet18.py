import os
from datetime import datetime

import keras

# import tensorflow as tf
# from classification_models.keras import Classifiers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPool2D,
)
from keras.losses import CategoricalCrossentropy
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from my_utils import checkpoint_exists, create_dataset, get_latest_epoch  # type:ignore

# ResNet18, preprocess_input = Classifiers.get("resnet18")
# hyperparams_AI = {
#     "batch_size": 128,
#     "epochs": 100,
#     "learning_rate": 2e-3,
#     "weight_decay": 5e-4,
#     "optimizer": "AdamW",
# }
# hyperparams_paper = {
#     "batch_size": 128,
#     "epochs": 200,
#     "learning_rate": 0.02,
#     "weight_decay": 5e-4,
#     "optimizer": "SGD",
# }
max_epochs = 50


# implementing residual blocks from scratch
@keras.saving.register_keras_serializable()
class ResidualLayer(keras.layers.Layer):
    def __init__(self, filters, first_stride=1, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.first_stride = first_stride

        self.conv_sequence = keras.Sequential(
            [
                Conv2D(filters, (3, 3), padding="same", strides=first_stride),
                BatchNormalization(),
                Activation("relu"),
                Conv2D(filters, (3, 3), padding="same"),
                BatchNormalization(),
            ]
        )

        # Skip connection handling
        self.skip_sequence = keras.Sequential()
        if first_stride != 1:
            self.skip_sequence.add(
                Conv2D(
                    filters, kernel_size=(1, 1), strides=first_stride, padding="same"
                )
            )

    def build(self, input_shape):
        super().build(input_shape)
        self.conv_sequence.build(input_shape)

    def call(self, inputs):
        x = self.conv_sequence(inputs)
        x_skip = (
            self.skip_sequence(inputs) if len(self.skip_sequence.layers) > 0 else inputs
        )
        x = Add()([x, x_skip])
        x = Activation("relu")(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters, "first_stride": self.first_stride})
        return config


def create_model(no_of_channels, no_of_subjects, h, T):
    fingerprinting_layers = keras.Sequential(
        [
            Input(shape=(h, T, no_of_channels)),
            Conv2D(64, (7, 7), strides=(2, 2), padding="same"),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="same"),
            ResidualLayer(64),
            ResidualLayer(64),
            ResidualLayer(128, first_stride=2),
            ResidualLayer(128),
            ResidualLayer(256, first_stride=2),
            ResidualLayer(256),
            ResidualLayer(512, first_stride=2),
            ResidualLayer(512),
            GlobalAveragePooling2D(),
            Flatten(),
        ]
    )
    # fingerprinting_layers.summary()

    id_layers = keras.Sequential(
        [
            Dense(1024, activation="relu"),
            Dropout(rate=0.3),
            Dense(no_of_subjects, activation="softmax"),
        ]
    )
    # id_layers.summary()

    model = keras.Sequential(
        [
            fingerprinting_layers,
            id_layers,
        ]
    )

    # Learning Rate Schedule
    # initial_learning_rate = 5e-3
    # lr_schedule = keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate,
    #     decay_steps=790,
    # )

    # Optimizer
    # optimizer = keras.optimizers.AdamW(
    #     learning_rate=lr_schedule,
    #     weight_decay=5e-4
    # )
    optimizer = keras.optimizers.SGD(
        learning_rate=0.02, momentum=0.9, weight_decay=5e-4
    )

    loss = CategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(),
        ],
    )
    return model


def create_model_old(no_of_subjects=90, h=20, T=160, trained=True):
    weights = "imagenet" if trained else ""
    base_model = ResNet18(  # noqa: F821 # type: ignore
        input_shape=(h, T, 3),
        weights=weights,
        classes=no_of_subjects,
        include_top=False,
    )

    fingerprint_layers = keras.Sequential(
        [
            base_model,
            keras.layers.Flatten(),
        ]
    )

    id_layers = keras.Sequential(
        [
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dropout(rate=0.25),
            keras.layers.Dense(no_of_subjects, activation="softmax"),
        ]
    )

    model = keras.Sequential(
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
            keras.metrics.CategoricalAccuracy(),
            # keras.metrics.Precision(),
            # keras.metrics.Recall(),
        ],
    )

    return model


# custom callback that deletes saved model of 2nd previous epoch
class DeleteCheckpointCallback(keras.callbacks.Callback):
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

    def on_train_end(self, logs=None):
        # delete all checkpoints after training ends
        for filename in os.listdir(self.save_file_dir):
            if filename.startswith(
                f"resnet18_C_{self.chosen_channels}-epoch_"
            ) and filename.endswith(".keras"):
                checkpoint_path = os.path.join(self.save_file_dir, filename)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)


def prepare_callbacks(save_file_dir, chosen_channels):
    # create save file directory if it doesn't exist
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
        print(f"Created directory {save_file_dir}")
    callbacks = []

    # set up checkpoint callback with the given save file directory
    cp_file_name = f"resnet18_C_{chosen_channels}" + "-epoch_{epoch:02d}.keras"
    checkpoint_path = os.path.join(save_file_dir, cp_file_name)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=False, verbose=1
    )
    callbacks.append(checkpoint_callback)

    del_checkpoint_callback = DeleteCheckpointCallback(
        save_file_dir, chosen_channels, max_epochs=max_epochs
    )
    callbacks.append(del_checkpoint_callback)

    # set up early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

    # set up profiling
    # tensorboard_callback = keras.callbacks.TensorBoard(
    #     log_dir=os.path.join(save_file_dir, "logs"),
    #     histogram_freq=1,
    #     write_steps_per_second=True,
    # )

    return callbacks


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

        # set up callbacks
        callbacks = prepare_callbacks(save_file_dir, chosen_channels)

        if not checkpoint_exists(save_file_dir, chosen_channels):
            # if no existing checkpoints are found, create the model from scratch and set initial epoch to 0
            print("no existing checkpoints found")
            model = create_model(
                no_of_subjects=no_of_subjects,
                h=h,
                T=T,
                no_of_channels=len(chosen_channels),
            )
            latest_epoch = 0
        else:
            # if we find checkpoints, pick up where we left off
            latest_epoch, model_dir = get_latest_epoch(save_file_dir, chosen_channels)
            from keras.models import load_model

            model = load_model(model_dir)
            print(f"loaded model from {model_dir}")
            print(f"resuming training from epoch {latest_epoch + 1}")

        stats = model.fit(
            x=dataset_train,
            validation_data=dataset_test,
            epochs=max_epochs,
            initial_epoch=latest_epoch,
            steps_per_epoch=None,
            validation_steps=None,
            verbose=1,
            callbacks=callbacks,
        )
    # pdb.set_trace()
    stopped_epoch = stats.epoch[-1] + 1 if hasattr(stats, "epoch") else None
    now = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")
    return {
        "datetime": now,
        "stopped_epoch": stopped_epoch,
        "categorical_accuracy": stats.history["categorical_accuracy"][-1],
        "loss": stats.history["loss"][-1],
        "val_categorical_accuracy": stats.history["val_categorical_accuracy"][-1],
        "val_loss": stats.history["val_loss"][-1],
    }


# ResNet18(input_shape=(20,160,3), weights='imagenet', include_top=False).summary()
# create_model().summary()
