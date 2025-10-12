import os
from datetime import datetime

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import CategoricalCrossentropy
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

from my_utils import checkpoint_exists, create_dataset, get_latest_epoch  # type:ignore

max_epochs = 50


def create_model(no_of_channels, no_of_subjects, h, T):
    fingerprinting_layers = keras.Sequential(
        [
            keras.layers.Input(shape=(h, T, no_of_channels)),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(512, (3, 3), activation="relu", padding="same"),
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
            fingerprinting_layers,
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
                f"CNN_C_{self.chosen_channels}-epoch_{epoch - 1:02d}.keras",
            )
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

    def on_train_end(self, logs=None):
        # delete all checkpoints after training ends
        for filename in os.listdir(self.save_file_dir):
            if "epoch" in filename and filename.endswith(".keras"):
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
    cp_file_name = f"CNN_C_{chosen_channels}" + "-epoch_{epoch:02d}.keras"
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
    #     dataset_train = create_dataset(X_train, y_train, sliding_window_out, strategy)
    #     dataset_test = create_dataset(X_test, y_test, sliding_window_out, strategy)
    #     # create save file directory if it doesn't exist
    #     if not os.path.exists(save_file_dir):
    #         os.mkdir(save_file_dir)
    #         print(f"Created directory {save_file_dir}")

    #     # set up checkpoint callback with the given save file directory
    #     cp_file_name = f"C_{chosen_channels}" + "-epoch_{epoch:02d}.keras"
    #     checkpoint_path = os.path.join(save_file_dir, cp_file_name)
    #     checkpoint_callback = ModelCheckpoint(
    #         filepath=checkpoint_path, save_weights_only=False, verbose=1
    #     )
    #     del_checkpoint_callback = DeleteCheckpointCallback(
    #         save_file_dir, chosen_channels, max_epochs=30
    #     )
    #     # checkpoint_callback = BackupAndRestore(
    #     #     backup_dir=checkpoint_path, save_freq="epoch", delete_checkpoint=True
    #     # )

    #     if not checkpoint_exists(save_file_dir, chosen_channels):
    #         # if no existing checkpoints are found, create the model from scratch and set initial epoch to 0
    #         print("no existing checkpoints found")
    #         model = create_model(no_of_channels, no_of_subjects, h, T)
    #         latest_epoch = 0
    #     else:
    #         # if we find checkpoints, pick up where we left off
    #         latest_epoch, model_dir = get_latest_epoch(save_file_dir, chosen_channels)
    #         model = keras.models.load_model(model_dir)
    #         print(f"loaded model from {model_dir}")
    #         print(f"resuming training from epoch {latest_epoch + 1}")

    #     stats = model.fit(
    #         x=dataset_train,
    #         validation_data=dataset_test,
    #         epochs=30,
    #         initial_epoch=latest_epoch,
    #         steps_per_epoch=None,
    #         validation_steps=None,
    #         verbose=1,
    #         callbacks=[checkpoint_callback, del_checkpoint_callback],
    #     )
    # # pdb.set_trace()
    # return stats.history["categorical_accuracy"][-1]
