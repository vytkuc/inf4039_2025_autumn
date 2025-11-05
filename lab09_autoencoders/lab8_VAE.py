import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
latent_dim   = 2
num_classes  = 10
input_shape  = (28, 28, 1)
use_bce_loss = False  # set True to use binary_crossentropy instead of MSE

# ---------------------------------------------------------------------
# Encoder: inputs are (image, one-hot label)
# ---------------------------------------------------------------------
image_input = Input(shape=input_shape, name="image_input")
label_input = Input(shape=(num_classes,), name="label_input")

# expand label to image shape and concatenate with image
label_dense    = layers.Dense(np.prod(input_shape), activation="relu")(label_input)
label_reshaped = layers.Reshape(input_shape)(label_dense)
enc_in = layers.Concatenate(axis=-1)([image_input, label_reshaped])

x = layers.Conv2D(32, 3, activation="relu", padding="same")(enc_in)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D(2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)

z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

def sampling(args):
    z_m, z_lv = args
    eps = tf.random.normal(shape=tf.shape(z_m))
    return z_m + tf.exp(0.5 * z_lv) * eps

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ---------------------------------------------------------------------
# Decoder: inputs are (z, one-hot label)
# ---------------------------------------------------------------------
z_in         = Input(shape=(latent_dim,), name="z_in")
decoder_lbl  = Input(shape=(num_classes,), name="decoder_label")
dec_concat   = layers.Concatenate(axis=-1)([z_in, decoder_lbl])

y = layers.Dense(7 * 7 * 64, activation="relu")(dec_concat)
y = layers.Reshape((7, 7, 64))(y)
y = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(y)
y = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
# use sigmoid for [0,1] images
decoded = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="decoded")(y)

decoder = Model([z_in, decoder_lbl], decoded, name="decoder")

# ---------------------------------------------------------------------
# Custom Model to compute loss in train_step (no KerasTensor misuse)
# ---------------------------------------------------------------------
class CVAE(Model):
    def __init__(self, encoder, decoder, use_bce=False, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.use_bce = use_bce
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker    = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # data = ([images, one_hot_labels], None) or ([images, one_hot_labels],)
        if isinstance(data, tuple) or isinstance(data, list):
            (x, y_onehot) = data[0]
        else:
            x, y_onehot = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, y_onehot], training=True)
            x_hat = self.decoder([z, y_onehot], training=True)

            # reconstruction loss
            if self.use_bce:
                # sum over pixels, mean over batch
                recon = tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, x_hat), axis=[1, 2, 3]
                )
            else:
                # MSE: sum over pixels, mean over batch
                recon = tf.reduce_sum(tf.math.squared_difference(x, x_hat), axis=[1, 2, 3])

            recon_loss = tf.reduce_mean(recon)

            # KL loss (per batch mean)
            kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            kl_loss = tf.reduce_mean(kl)

            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            (x, y_onehot) = data[0]
        else:
            x, y_onehot = data

        z_mean, z_log_var, z = self.encoder([x, y_onehot], training=False)
        x_hat = self.decoder([z, y_onehot], training=False)

        if self.use_bce:
            recon = tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x, x_hat), axis=[1, 2, 3]
            )
        else:
            recon = tf.reduce_sum(tf.math.squared_difference(x, x_hat), axis=[1, 2, 3])

        recon_loss = tf.reduce_mean(recon)
        kl = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        kl_loss = tf.reduce_mean(kl)
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

# Build CVAE
cvae = CVAE(encoder, decoder, use_bce=use_bce_loss, name="cvae")
cvae.compile(optimizer=tf.keras.optimizers.Adam())

# ---------------------------------------------------------------------
# Data: MNIST (demo)
# ---------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]
x_test  = (x_test.astype("float32")  / 255.0)[..., None]
y_train_oh = to_categorical(y_train, num_classes)
y_test_oh  = to_categorical(y_test,  num_classes)

# Train
cvae.fit(
    x=(x_train, y_train_oh),
    epochs=20,
    batch_size=128,
    validation_data=((x_test, y_test_oh), None),
    verbose=1
)

# ---------------------------------------------------------------------
# Sample: pick a label, sample z ~ N(0,1), decode
# ---------------------------------------------------------------------
import matplotlib.pyplot as plt

def generate(label_int, n=10):
    lbl = to_categorical([label_int], num_classes)
    for _ in range(n):
        z_sample = np.random.normal(size=(1, latent_dim)).astype("float32")
        img = decoder.predict([z_sample, lbl], verbose=0)[0, ..., 0]
        plt.imshow(img, cmap="gray")
        plt.title(f"Generated label = {label_int}")
        plt.axis("off")
        plt.show()

generate(7, n=3)
