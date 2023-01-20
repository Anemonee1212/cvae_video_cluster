# Dense CVAE model.
# Using more complex Convolutional Variational Autoencoder with Dense Neural Network layers
# to reduce the dimensionality of video frame images before performing Gaussian Mixture Model clustering.
import math
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import tensorflow as tf

# ========== Hyper-parameters ==========
__author__ = ["Tiancheng (Robert) Shi"]
batch_size = 16
epochs = 300
img_height, img_width = (120, 200)
latent_layer_dim = 1000
lr = 1e-4
num_examples_to_generate = 16
seed = 3407
source = "multi"

tf.random.set_seed(seed)

# ========== Load and preprocess dataset ==========
data_save_path = "output/" + source + "/"

if source == "multi":
    data_root = pathlib.Path("images")
    data_image_label = tf.keras.utils.image_dataset_from_directory(
        data_root, label_mode = "binary",
        batch_size = batch_size, image_size = (img_height, img_width)
    )
    data_image = data_image_label.map(lambda img, lab: img)
    data_label = data_image_label.map(lambda img, lab: lab)
else:
    data_root = pathlib.Path("images/" + source)
    data_image = tf.keras.utils.image_dataset_from_directory(
        data_root, label_mode = None,
        batch_size = batch_size, image_size = (img_height, img_width)
    )

for image_batch in data_image:
    print(image_batch.shape)
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.axis("off")
    plt.show()
    break

normalize_layer = tf.keras.layers.Rescaling(1 / 255)
data_norm = data_image.map(lambda img: normalize_layer(img))
# for norm_batch in data_norm:
#     img1 = norm_batch[0]
#     print(np.min(img1), np.max(img1))
#     break


# ========== Construct model ==========
class CVAE(tf.keras.Model):
    """
    Convolutional variational autoencoder with Dense Neural Network layers.
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
                filters = 32, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(
                filters = 64, kernel_size = 3, strides = 1,
                padding = "same", activation = "relu"
            ),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8000, activation = "relu"),
            tf.keras.layers.Dense(4000, activation = "relu"),
            # No activation
            tf.keras.layers.Dense(2 * latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape = (latent_dim, )),
            tf.keras.layers.Dense(4000, activation = "relu"),
            tf.keras.layers.Dense(8000, activation = "relu"),
            tf.keras.layers.Dense(64 * img_height * img_width // 64, activation = "relu"),
            tf.keras.layers.Reshape(target_shape = (img_height // 8, img_width // 8, 64)),
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
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits = 2, axis = 1)
        # axis = 1 indicates that the dense layer of (latent_dim * 2) neurons from encoder output
        # are divided into latent_dim for mean and latent_dim for variance
        return mean, log_var

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape = mean.shape)
        return eps * tf.exp(log_var * 0.5) + mean

    def decode(self, z):
        return self.decoder(z)


# ========== Defining loss function and helper functions ==========
adam_opt = tf.keras.optimizers.Adam(lr)
subplot_axis = math.ceil(math.sqrt(num_examples_to_generate))


def compute_loss(model, x):
    mean, log_var = model.encode(x)
    z = model.reparameterize(mean, log_var)
    x_recon = model.decode(z)
    image_mse = tf.metrics.mean_squared_error(y_true = x, y_pred = x_recon)
    return tf.reduce_mean(image_mse)


def reconstruct_images(model, images):
    mean, _ = model.encode(images)
    predictions = model.decode(mean)
    predictions = (predictions * 255).numpy().astype("uint8")
    return predictions, mean


def generate_test_images(model, epoch, test_images):
    predictions, _ = reconstruct_images(model, test_images)
    for i in range(predictions.shape[0]):
        plt.subplot(subplot_axis, subplot_axis, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis("off")

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(data_save_path + "test/epoch_{:03d}.png".format(epoch))
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


my_model = CVAE(latent_layer_dim)
loss_list = []

for i in range(epochs):
    mse = tf.keras.metrics.Mean()
    for norm_batch in data_norm:
        train_step(my_model, norm_batch, adam_opt)
        mse(compute_loss(my_model, norm_batch))

    mse_scaled = 255 * mse.result()
    print("Epoch: {}, MSE: {}".format(i, mse_scaled))
    loss_list.append(mse_scaled)
    if i % 10 == 0 or i == epochs - 1:
        generate_test_images(my_model, i, test_sample)

print("Session Terminated.")
plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(data_save_path + "loss_function.png")
plt.show()

# ========== Testing (output encoded data) ==========
encoded_tensor_batch = []
if source == "multi":
    label_batch = []
    for batch_idx, data_batch in enumerate(zip(data_norm, data_label)):
        label_batch.append(data_batch[1])

        img_recon, img_batch_encode = reconstruct_images(my_model, data_batch[0])
        encoded_tensor_batch.append(img_batch_encode)
        for i in range(img_recon.shape[0]):
            plt.imshow(img_recon[i, :, :, :])
            plt.axis("off")
            plt.savefig(data_save_path + "img_out/{:03d}.png".format(batch_idx * batch_size + i))

    label_array = tf.concat(label_batch, axis = 0)
    np.savetxt(data_save_path + "data_label.csv", label_array.numpy(), delimiter = ",")
else:
    for batch_idx, img_batch in enumerate(data_norm):
        img_recon, img_batch_encode = reconstruct_images(my_model, img_batch)
        encoded_tensor_batch.append(img_batch_encode)
        for i in range(img_recon.shape[0]):
            plt.imshow(img_recon[i, :, :, :])
            plt.axis("off")
            plt.savefig(data_save_path + "img_out/{:03d}.png".format(batch_idx * batch_size + i))

data_encoded = tf.concat(encoded_tensor_batch, axis = 0)
np.savetxt(data_save_path + "data.csv", data_encoded.numpy(), delimiter = ",")
