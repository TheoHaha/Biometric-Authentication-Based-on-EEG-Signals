import os

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Input
from keras.losses import CategoricalCrossentropy
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from my_utils import create_dataset
from tcn import TCN


def create_tcn_model(no_of_channels, no_of_subjects, h, T):
    inputs = Input(shape=(h, T, no_of_channels))
    # Reshape for TCN: (batch, time_steps, features)
    x = keras.layers.Reshape((T, h * no_of_channels))(inputs)
    x = TCN(
        nb_filters=64, kernel_size=8, dilations=[1, 2, 4, 8], return_sequences=False
    )(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(no_of_subjects, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )
    return model


# Custom callback to delete old checkpoints
class DeleteCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, save_file_dir, chosen_channels, max_epochs):
        self.save_file_dir = save_file_dir
        self.chosen_channels = chosen_channels
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            checkpoint_path = os.path.join(
                self.save_file_dir,
                f"tcn_C_{self.chosen_channels}-epoch_{epoch - 1:02d}.keras",
            )
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def on_train_end(self, logs=None):
        # delete all checkpoints after training ends
        for filename in os.listdir(self.save_file_dir):
            if filename.startswith(
                f"tcn_C_{self.chosen_channels}-epoch_"
            ) and filename.endswith(".keras"):
                checkpoint_path = os.path.join(self.save_file_dir, filename)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)


def train_tcn_on_V(
    V, labels, chosen_channels, save_file_dir, h, T, sliding_window_out, strategy
):
    no_of_channels = V.shape[1]
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
        cp_file_name = f"tcn_C_{chosen_channels}" + "-epoch_{epoch:02d}.keras"
        checkpoint_path = os.path.join(save_file_dir, cp_file_name)
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=False, verbose=1
        )
        del_checkpoint_callback = DeleteCheckpointCallback(
            save_file_dir, chosen_channels, max_epochs=30
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            start_from_epoch=5,
            patience=3,
            min_delta=0.0001,
            baseline=0.001,
            restore_best_weights=False,
            verbose=1,
        )

        # Always create a new model for TCN (no checkpoint resume logic here, but can be added)
        model = create_tcn_model(no_of_channels, no_of_subjects, h, T)

        stats = model.fit(
            x=dataset_train,
            validation_data=dataset_test,
            epochs=30,
            steps_per_epoch=None,
            validation_steps=None,
            verbose=1,
            callbacks=[
                checkpoint_callback,
                del_checkpoint_callback,
                early_stopping_callback,
            ],
        )
    return {
        "categorical_accuracy": stats.history["categorical_accuracy"][-1],
        "loss": stats.history["loss"][-1],
        "val_categorical_accuracy": stats.history["val_categorical_accuracy"][-1],
        "val_loss": stats.history["val_loss"][-1],
    }
