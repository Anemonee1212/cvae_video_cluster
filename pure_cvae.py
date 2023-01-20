# Re-implementing the method originally used by Omer Onder.
# Using Convolutional Variational Autoencoder to reduce the dimensionality of video frame images,
# before performing K-means clustering.
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf

# ========== Hyper-parameters ==========
__author__ = ["Omer Onder", "Tiancheng (Robert) Shi"]
batch_size = 32
epochs = 100
img_height = 240
img_width = 400
latent_filter_num = 128
lr = 1e-4
num_examples_to_generate = 16

# ========== Check images ==========
data_root = pathlib.Path("images/us")
img_save_path = "output/omer/"

# ========== Load and preprocess dataset ==========
data_image = tf.keras.utils.image_dataset_from_directory(
    data_root, label_mode = None,
    batch_size = batch_size, image_size = (img_height, img_width)
)

normalize_layer = tf.keras.layers.Rescaling(1 / 255)
data_norm = data_image.map(lambda img: normalize_layer(img))


# ========== Construct my_model ==========
class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder.
    """

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (img_height, img_width, 3)),
            tf.keras.layers.Conv2D(
                filters = 32, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(
                filters = 64, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(
                filters = 128, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            # No activation
            tf.keras.layers.Conv2D(filters = latent_dim * 2, kernel_size = 3, strides = 1, padding = "same")
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (img_height // 8, img_width // 8, latent_dim)),
            tf.keras.layers.Conv2DTranspose(
                filters = 128, kernel_size = 3, strides = 2,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.Conv2D(
                filters = 128, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters = 64, kernel_size = 3, strides = 2,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.Conv2D(
                filters = 64, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.Conv2DTranspose(
                filters = 32, kernel_size = 3, strides = 2,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.Conv2D(
                filters = 32, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            # No activation
            tf.keras.layers.Conv2DTranspose(filters = 3, kernel_size = 3, strides = 1, padding = "same")
        ])

    @tf.function
    def sample(self, eps = None):
        if eps is None:
            eps = tf.random.normal(shape = (100, self.latent_dim))

        return self.decode(eps)

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 3)
        # axis = 3 indicates that the (latent_dim * 2) filters from encoder output
        # are divided into latent_dim for mean and latent_dim for variance
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape = mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)


# ========== Defining mse function and helper functions ==========
optimizer = tf.keras.optimizers.Adam(lr)


def compute_loss(model, x):
    mean, log_var = model.encode(x)
    z = model.reparameterize(mean, log_var)
    x_recon = model.decode(z)
    mse = tf.metrics.mean_squared_error(y_true = x, y_pred = x_recon)
    return tf.reduce_mean(mse)


def generate_and_save_images(model, epoch, test_sample):
    mean, log_var = model.encode(test_sample)
    z = model.reparameterize(mean, log_var)
    predictions = model.sample(z)
    predictions = (predictions * 255).numpy().astype("uint8")

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(img_save_path + "image_at_epoch_{:04d}.png".format(epoch))
    plt.show()


# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
test_sample = None
for test_batch in data_norm.take(1):
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]


# ========== Training Session ==========
@tf.function
def train_step(model, x, optimizer):
    """
    Executes one training step and returns the mse.

    This function computes the mse and gradients, and uses the latter to
    update the my_model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


model = CVAE(latent_filter_num)
loss_list = []

for epoch in range(epochs):
    loss = tf.keras.metrics.Mean()
    for norm_batch in data_norm:
        train_step(model, norm_batch, optimizer)
        loss(compute_loss(model, norm_batch))

    mse_scaled = 255 * loss.result()
    print("Epoch: {}, MSE: {}".format(epoch, mse_scaled))
    loss_list.append(mse_scaled)
    if epoch % 10 == 0:
        generate_and_save_images(model, epoch, test_sample)

print("Session Terminated.")
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(img_save_path + "loss_function.png")
plt.show()
