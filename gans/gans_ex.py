
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.minst import input_data

print("Downloading mnist data . . .")
mnist = input_data.read.data_sets("MNIST_data/")
mnist_image = mnist.train.image
mnist_labels = mnist.train.labels
print("Finished downloading.")

r = 0
def next_batch(data, size):
    global r
    if r*size*size > len(data):
        r = 0
    x_train_batch = data[size*r : r*size+size, :]
    r = r + 1
    return x_train_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))

def init_bias(shape):
    return tf.Variable(tf.constant(0.2, shape=shape))

class Generator:
    def __init__(self):
        with tf.variable_scope('g'):
            self.gW1 = init_weights([100, 256])
            self.gb1 = init_bias([256])
            self.gW2 = init_weights([256, 784])
            self.gb2 = init_bias([784])

    def forward(self, z, training=True):
        fc1 = tf.matmul(z, self.gW1) + self.gb1
        fc1 = tf.layers.batch_normalization(fc1, training=training)
        fc1 = tf.nn.leaky_relu(fc1)
        fc2 = tf.nn.sigmoid(tf.matmul(fc1, self.gW2) + self.gb2)
        return fc2

class Discriminator:
    def __init__(self):
        with tf.variable_scope('d'):
            self.dW1 = init_weights([5,5,1,16])
            self.db1 = init_bias([16])
            self.dW2 = init_weights([3,3,16,32])
            self.db2 = init_bias([32])

            self.W3 = init_weights([1568, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([128, 1])
            self.b4 = init_bias([1])

    def forward(self, X):
        self.X = tf.reshape(X, shape=[-1, 28, 28, 1])
        conv1 = tf.nn.leaky_relu(tf.nn.conv2d(self.X, self.dW1, strides=[1, 2, 2, 1], padding = 'SAME') + self.db1)
        conv1 = tf.layers.batch_normalization(conv1, True)
        conv2 = tf.nn.leaky_relu(tf.nn.conv2d(conv1, self.dW2, strides=[1, 2, 2, 1], padding='SAME')+ self.db2)
        conv2 = tf.layers.batch_normalization(conv2, True)
        conv2 = tf.reshape(conv2, shape[-1, 7*7*32])

        fc1 = tf.nn.leaky_relu(tf.matmul(conv2, self.W3) + self.b3)
        logits = tf.matmul(fc1, self.W4) + self.b4
        fc2 = tf.nn.sigmoid(logits)

        return fc2, logits

def cost(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

d = Discriminator()
g = Generator()

phX = tf.placeholder(tf.float32, [None, 784])
phZ = tf.placeholder(tf.float32, [None, 100])

G_out = g.forward(phZ)
G_out_sample = g.forward(phZ, False)

D_out_real, D_logits_real = d.forward(phX)
D_out_fake, D_logits_fake = d.forward(G_out)

D_real_loss = cost(D_logits_real, tf.ones_like(D_logits_real))
D_fake_loss = cost(D_logits_fake, tf.zeros_like(D_logits_fake))

D_loss = D_real_loss + D_fake_loss
G_loss = cost(D_logits_fake, tf.ones_like(D_logits_fake))

lr = 0.001

epochs = 7000

pretrain_epochs = 4000
batch_size = 50

train_vars = tf.trainable_variables()

dvars = [var for var in train_vars if 'd' in var.name]
gvars = [var for var in train_vars if 'g' in var.name]

D_train = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=dvars)
G_train = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=gvars)

init = tf.global_variables_initializer()

if not os.path.exists('generated_images/'):
    os.makedirs('generated_images/')

with tf.Session() as sess:
    for i in range(10):
        sess.run(init)

        k = 0
        l = 10
        data = mnist_images[mnist_labels == i]
        print(f"Starting training for label {i} . . .")
        g_cost = []
        d_cost = []

        for j in range(epochs):

            batch_X = next_batch(data, batch_size)

            batch_z = np.random.randn(batch_size, 100)

            _, d_loss = sess.run([D_train, D_loss], feed_dict={phX:batch_X, phZ: batch_z})

            _, g_loss = sess.run([G_train, G_loss], feed_dict = {phZ:batch_z})

            d_cost.append(d_loss)
            g_cost.append(g_loss)

            if j % pretrain_epochs//10 == 0 and j < pretrain_epochs:
                print(f"Pretraining. Generating images for label {i} in {l}. ")
                l = l - 1

            if j % 100 == 0 and j >= pretrain_epochs:
                sample_z = np.random.randn(1,100)

                gen_sample = sess.run(G_out_sample, feed_dict={phZ:sample_z})

                print(f"Iteration {j}. G_loss {g_loss}. D_loss {d_loss}")

                image = plt.imshow(gen_sample.reshape(28, 28), cmap="Greys_r")
                plt.savefig(f"generated_images/sample{i}_{k+1}.png")
                k += 1

