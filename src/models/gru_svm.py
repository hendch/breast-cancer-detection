import tensorflow as tf
from tensorflow.keras import layers, models


def build_gru_svm(input_shape, config):
    inputs = layers.Input(shape=input_shape)

    x = layers.GRU(
        units=config["gru_units"],
        activation="tanh",
        recurrent_activation="sigmoid",
    )(inputs)

    x = layers.Dropout(config["dropout"])(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss=tf.keras.losses.SquaredHinge(),
        metrics=["accuracy"],
    )

    return model
