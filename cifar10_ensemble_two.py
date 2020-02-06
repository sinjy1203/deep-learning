import tensorflow as tf
import numpy as np
from keras.datasets import cifar10

tf.set_random_seed(777)

with tf.device('/device:GPU:0'):
    class Model:
        def __init__(self, sess, name):
            self.sess = sess
            self.name = name
            self.build_net()

        def build_net(self):
            with tf.variable_scope(self.name):
                self.x = tf.placeholder(tf.float32, [None, 32, 32, 3])
                self.y = tf.placeholder(tf.float32, [None, 10])
                self.keep_prob = tf.placeholder(tf.float32)

                W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
                L1 = tf.nn.conv2d(self.x, W1, strides=[1, 1, 1, 1], padding='SAME')
                L1 = tf.nn.relu(L1)
                L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

                W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
                L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
                L2 = tf.nn.relu(L2)
                L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)

                W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
                L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
                L3 = tf.nn.relu(L3)
                L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
                L3 = tf.reshape(L3, [-1, 4 * 4 * 128])

                W4 = tf.get_variable("W4", shape=[4 * 4 * 128, 625], initializer=tf.contrib.layers.xavier_initializer())
                b4 = tf.Variable(tf.random_normal([625]))
                L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
                L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)

                W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
                b5 = tf.Variable(tf.random_normal([10]))
                self.logits = tf.matmul(L4, W5) + b5

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1)), tf.float32))

        def predict(self, x_test, keep_prob=1):
            return self.sess.run(self.logits, feed_dict={self.x: x_test, self.keep_prob: keep_prob})

        def get_accuracy(self, x_test, y_test, keep_prob=1):
            return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prob})

        def train(self, x_train, y_train, keep_prob=0.7):
            return self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.x: x_train, self.y: y_train, self.keep_prob: keep_prob})


    def training(model_number):
        print(model_number, "번째 트레이닝")
        for epochs in range(total_epochs):
            avg_cost = 0
            for i in range(number):
                xs = x_train[i * batch_size:(i + 1) * batch_size]
                ys = y_train[i * batch_size:(i + 1) * batch_size]

                c, _ = models[model_number].train(xs, ys)
                avg_cost += c / number
            print(epochs + 1, "   ", avg_cost)
        accuracy_list[model_number] = models[model_number].get_accuracy(x_test, y_test)
        print("정확도: ", accuracy_list[model_number])


    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np.squeeze(y_train)
    # print(y_train.shape)
    y_train = np.eye(10)[y_train]
    # print(y_train.shape)
    y_test = np.squeeze(y_test)
    y_test = np.eye(10)[y_test]

    sess = tf.Session()
    models = []
    models_num = 5
    accuracy_list = np.zeros(models_num)
    for m in range(models_num):
        models.append(Model(sess, "model" + str(m)))
    sess.run(tf.global_variables_initializer())

    total_epochs = 40
    batch_size = 500
    number = int(50000 / batch_size)

    for j in range(models_num):
        training(j)

    print("각각의 정확도: ", accuracy_list)

    print("--------------")
    predict_sum = np.zeros([len(y_test), 10])
    for m_idx, m in enumerate(models):
        predict_sum += m.predict(x_test)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predict_sum, 1), tf.argmax(y_test, 1)), tf.float32))))

