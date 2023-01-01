import datetime
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorboard.plugins.hparams import api as hp


def generate_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(base_dir, image_size=(200, 200), subset='training',
                                                                seed=1,
                                                                validation_split=0.1, batch_size=32)

    test_dataset = tf.keras.utils.image_dataset_from_directory(base_dir, image_size=(200, 200), subset='validation',
                                                               seed=1,
                                                               validation_split=0.1, batch_size=32)
    return train_dataset, test_dataset


def generate_checkpoint_callback(hparams):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath="Checkpoints/" + "dropout=" + str(hparams[HP_DROPOUT]),
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_freq='epoch')


def build_model(hparams):
    return tf.keras.models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(hparams[HP_DROPOUT]),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(hparams[HP_DROPOUT]),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])


def run_training(train_dataset, test_dataset, hparams):
    model = build_model(hparams)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    log_dir = "logs/fit/" + "dropout=" + str(hparams[HP_DROPOUT])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = generate_checkpoint_callback(hparams)
    return model.fit(train_dataset, epochs=10, validation_data=test_dataset,
                     callbacks=[tensorboard_callback, model_checkpoint_callback])


base_dir = 'dataset'
train_dataset, test_dataset = generate_datasets()
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2]))
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=[HP_DROPOUT], metrics=[hp.Metric("accuracy", display_name='Accuracy')])

for dropout_rate in HP_DROPOUT.domain.values:
    hparams = {
        HP_DROPOUT: dropout_rate
    }
    history = run_training(train_dataset, test_dataset, hparams)
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.show()
