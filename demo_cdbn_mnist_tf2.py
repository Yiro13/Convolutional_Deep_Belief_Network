import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import cdbn_backup as cdbn


""" --------------------------------------------
    ------------------- DATA -------------------
    -------------------------------------------- """


class MNIST_HANDLER(object):
    def __init__(self):
        (
            (self.training_data, self.training_labels),
            (self.test_data, self.test_labels),
        ) = mnist.load_data()
        self.num_training_example = self.training_data.shape[0]
        self.num_test_example = self.test_data.shape[0]

        self.training_data = (
            self.training_data.reshape(-1, 28 * 28).astype("float32") / 255.0
        )
        self.test_data = self.test_data.reshape(-1, 28 * 28).astype("float32") / 255.0

        self.training_labels = to_categorical(self.training_labels, num_classes=10)
        self.test_labels = to_categorical(self.test_labels, num_classes=10)

    def do_whiten(self):
        self.whiten = True
        data_to_be_whitened = np.copy(self.training_data)
        mean = np.sum(data_to_be_whitened, axis=0) / self.num_training_example
        mean = np.tile(mean, self.num_training_example)
        mean = np.reshape(mean, (self.num_training_example, 784))
        centered_data = data_to_be_whitened - mean
        covariance = np.dot(centered_data.T, centered_data) / self.num_training_example
        U, S, V = np.linalg.svd(covariance)
        epsilon = 1e-5
        lambda_square = np.diag(1.0 / np.sqrt(S + epsilon))
        self.whitening_mat = np.dot(np.dot(U, lambda_square), V)
        self.whitened_training_data = np.dot(centered_data, self.whitening_mat)

        data_to_be_whitened = np.copy(self.test_data)
        mean = np.sum(data_to_be_whitened, axis=0) / self.num_test_example
        mean = np.tile(mean, self.num_test_example)
        mean = np.reshape(mean, (self.num_test_example, 784))
        centered_data = data_to_be_whitened - mean
        self.whitened_test_data = np.dot(centered_data, self.whitening_mat)

    def next_batch(self, batch_size, type="train"):
        if type == "train":
            start = (self.training_index * batch_size) % self.num_training_example
            end = start + batch_size
            return self.training_data[start:end], self.training_labels[start:end]
        elif type == "test":
            start = (self.test_index * batch_size) % self.num_test_example
            end = start + batch_size
            return self.test_data[start:end], self.test_labels[start:end]


mnist_dataset = MNIST_HANDLER()
# mnist_dataset.do_whiten()
# sess = tf.Session()


""" ---------------------------------------------
    ------------------- MODEL -------------------
    --------------------------------------------- """

my_cdbn = cdbn.CDBN(
    "mnist_cdbn",
    20,
    "/home/arthur/pedestrian_detection/log",
    mnist_dataset,
    sess,
    verbosity=2,
)

my_cdbn.add_layer(
    "layer_1",
    fully_connected=False,
    v_height=28,
    v_width=28,
    v_channels=1,
    f_height=11,
    f_width=11,
    f_number=40,
    init_biases_H=-3,
    init_biases_V=0.01,
    init_weight_stddev=0.01,
    gaussian_unit=True,
    gaussian_variance=0.2,
    prob_maxpooling=True,
    padding=True,
    learning_rate=0.00005,
    learning_rate_decay=0.5,
    momentum=0.9,
    decay_step=50000,
    weight_decay=2.0,
    sparsity_target=0.003,
    sparsity_coef=0.1,
)

my_cdbn.add_layer(
    "layer_2",
    fully_connected=False,
    v_height=14,
    v_width=14,
    v_channels=40,
    f_height=7,
    f_width=7,
    f_number=40,
    init_biases_H=-3,
    init_biases_V=0.025,
    init_weight_stddev=0.025,
    gaussian_unit=False,
    gaussian_variance=0.2,
    prob_maxpooling=True,
    padding=True,
    learning_rate=0.0025,
    learning_rate_decay=0.5,
    momentum=0.9,
    decay_step=50000,
    weight_decay=0.1,
    sparsity_target=0.1,
    sparsity_coef=0.1,
)

my_cdbn.add_layer(
    "layer_3",
    fully_connected=True,
    v_height=1,
    v_width=1,
    v_channels=40 * 7 * 7,
    f_height=1,
    f_width=1,
    f_number=200,
    init_biases_H=-3,
    init_biases_V=0.025,
    init_weight_stddev=0.025,
    gaussian_unit=False,
    gaussian_variance=0.2,
    prob_maxpooling=False,
    padding=False,
    learning_rate=0.0025,
    learning_rate_decay=0.5,
    momentum=0.9,
    decay_step=50000,
    weight_decay=0.1,
    sparsity_target=0.1,
    sparsity_coef=0.1,
)

my_cdbn.add_softmax_layer(10, 0.1)

my_cdbn.lock_cdbn()


""" ---------------------------------------------
    ------------------ TRAINING -----------------
    --------------------------------------------- """
my_cdbn.manage_layers(
    ["layer_1", "layer_2", "layer_3"],
    [],
    [10000, 10000, 10000],
    [1, 1, 1],
    20000,
    restore_softmax=False,
    fine_tune=True,
)
my_cdbn.do_eval()
