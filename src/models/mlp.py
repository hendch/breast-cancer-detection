import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp(input_dim, config):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for units in config["hidden_layers"]:
        model.add(layers.Dense(units, activation="relu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model
