import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorboard.plugins.hparams import api as hp

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def generate_datasets():
    DATASET_SIZE = 830
    base_dir = 'dataset'
    dataset = tf.keras.utils.image_dataset_from_directory(base_dir, image_size=(200, 200), seed=1, batch_size=32)
    #dataset = dataset.concatenate(dataset.map(augment_data))
    train_dataset = dataset.take(int(DATASET_SIZE * 0.9))
    val_dataset = dataset.skip(int(DATASET_SIZE * 0.9))

    return train_dataset, val_dataset


def augment_data(image, label):
    flip = tf.keras.layers.RandomFlip("vertical")
    rotate = tf.keras.layers.RandomRotation(0.2)
    translation = tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
    brightness = tf.keras.layers.RandomBrightness(0.3)

    return flip(rotate(translation(brightness(image)))), label


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
        layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        layers.Dropout(hparams[HP_DROPOUT]),
        layers.BatchNormalization(),
        layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        layers.Dropout(hparams[HP_DROPOUT]),
        layers.BatchNormalization(),
        layers.Dense(1, activation='sigmoid')
    ])


def run_training(train_dataset, test_dataset, hparams):
    log_dir = "logs/fit/" + "dropout=" + str(hparams[HP_DROPOUT]) + "num_units" + str(hparams[HP_NUM_UNITS])
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)
    model = build_model(hparams)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint_callback = generate_checkpoint_callback(hparams)
    return model.fit(train_dataset, epochs=20, validation_data=test_dataset,
                     callbacks=[tensorboard_callback, model_checkpoint_callback])


train_dataset, test_dataset = generate_datasets()
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([512, 1024]))
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(hparams=[HP_NUM_UNITS, HP_DROPOUT], metrics=[hp.Metric("accuracy", display_name='Accuracy')])

for dropout_rate in HP_DROPOUT.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        hparams = {
            HP_DROPOUT: dropout_rate,
            HP_NUM_UNITS: num_units
        }
        history = run_training(train_dataset, test_dataset, hparams)
        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot()
        history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
        plt.show()
