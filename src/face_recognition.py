import inception_resnet_v2
import os
import numpy as np
import tensorflow as tf
import cv2
from random import shuffle


# parameters
IMG_H, IMG_W, IMG_C = 250, 250, 3
embedding_size = 512
num_class = 300
learning_rate = 0.001
regularization_factor = 0.01


class loader():
    def __init__(self, dir_name, img_h, img_w):
        label = 0
        self.index = 0
        self.batch_list = []
        #self.cnt = 0
        self.epoch = 0
        self.steps = 0
        self.img_h, self.img_w = img_h, img_w
        for root, dirs, files in os.walk(dir_name):
            if files:
                for file_name in files:
                    path = os.path.join(root, file_name)
                    self.batch_list.append([path, label])
                label += 1

        self.size = len(self.batch_list)
        shuffle(self.batch_list)

    def batch(self, n):
        batch_image, batch_label = [], []
        for i in range(n):
            if self.index >= self.size:
                shuffle(self.batch_list)
                self.index = 0

            image = cv2.imread(self.batch_list[self.index][0])
            image = cv2.resize(image, (self.img_w, self.img_h))
            batch_image.append(image)
            batch_label.append(self.batch_list[self.index][1])
            self.index += 1

        return batch_image, batch_label


# graph
with tf.Graph().as_default() as g:
    image_placeholder = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C], name='image')
    label_placeholder = tf.placeholder(tf.int32, [None], name='label')
    istrain_placeholder = tf.placeholder(tf.bool)
    dropout_placeholder = tf.placeholder(tf.float32)

    embedding, end_points = inception_resnet_v2.inception_resnet_v2(image_placeholder, istrain_placeholder,
                                                                    dropout_placeholder, embedding_size)
    prelogit = tf.layers.dense(embedding, num_class)

    with tf.name_scope('loss'):
        softmax_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_placeholder, logits=prelogit))
        l2_regularization_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.get_collection('trainable_variables')])
        total_loss = softmax_loss + regularization_factor * l2_regularization_loss

    with tf.name_scope('optimizer'):
        opt = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    with tf.name_scope('saver'):
        saver = tf.train.Saver(max_to_keep=1)


dir_image = 'dataset'
dir_model = 'train_model'
data_loader = loader(dir_image, IMG_H, IMG_W)

EPOCH = 40
BATCH_SIZE = 32
TRAIN_SIZE = 631
STEPS = TRAIN_SIZE // BATCH_SIZE


# training part
with tf.Session(graph=g) as sess:
    checkpoint = tf.train.latest_checkpoint(dir_model)
    if checkpoint:
        saver.restore(sess, checkpoint)
    else:
        sess.run(tf.global_variables_initializer())

    while data_loader.epoch < EPOCH:
        for i in range(STEPS):
            batch_image, batch_label = data_loader.batch(BATCH_SIZE)
            feed_dict = {
                image_placeholder : batch_image,
                label_placeholder : batch_label,
                istrain_placeholder : True,
                dropout_placeholder : 0.8
            }
            _, loss = sess.run([opt, total_loss], feed_dict=feed_dict)
            print('epoch: {}, steps: {}, train-loss: {}'.format(data_loader.epoch+1, i+1, loss))
        data_loader.epoch += 1
    saver.save(sess, os.path.join(dir_model, 'model.ckpt'))


# test part
def test(face1, face2):
    feed_dict = {
        image_placeholder: [face1, face2],
        istrain_placeholder: False,
        dropout_placeholder: 1
    }
    embedding_vector = sess.run(embedding, feed_dict=feed_dict)

    # inner_product calculation
    prod_sum = np.sum(np.multiply(embedding_vector[0], embedding_vector[1]))
    norm0 = np.sqrt(np.sum(embedding_vector[0] ** 2))
    norm1 = np.sqrt(np.sum(embedding_vector[1] ** 2))
    inner_product = prod_sum / norm0 / norm1
    print(inner_product)

with tf.Session(graph=g) as sess:
    checkpoint = tf.train.latest_checkpoint(dir_model)
    saver.restore(sess, checkpoint)

    face1 = cv2.resize(cv2.imread('test_01.jpg'), (IMG_W, IMG_H))
    face2 = cv2.resize(cv2.imread('test_02.jpg'), (IMG_W, IMG_H))
    face3 = cv2.resize(cv2.imread('test_03.jpg'), (IMG_W, IMG_H))
    test(face1, face2)
    test(face1, face3)



