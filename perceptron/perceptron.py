import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plane_function(x, y):
    return x + 2 * y

def data_generator(num):
    # x [-100, 100]
    # y [-100, 100]
    # x + 2y + 4 = 0
    datas = list()
    labels = list()
    for _ in range(num):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        if plane_function(x, y) == 0:
            continue
        datas.append([x, y])
        labels.append(1 if plane_function(x, y) > 0 else -1)
    datas = np.array(datas)
    labels = np.array(labels)
    labels = labels.reshape([labels.shape[0], 1])
    return np.array(datas), np.array(labels, dtype=np.float32)

def wrong_num(labels, datas, model):
    num = 0
    pred = model(datas)
    for x, y in zip(labels, pred):
        x = x[0]
        y = y.numpy()[0]
        if x * y < 0:
            num += 1
    return num

class Perceptron(tf.keras.Model):
    def __init__(self, num=2):
        super(Perceptron, self).__init__()
        self.w = tf.Variable(tf.random.normal([2, 1]))

    def call(self, x):
        pred = tf.matmul(x, self.w)
        return pred

def loss_func(y, pred_y):
    y = tf.transpose(y)
    return tf.reduce_sum(-1 * tf.matmul(y, pred_y))

@tf.function
def train_step(data, labels, model):
    with tf.GradientTape() as tape:
        pred = model(data)
        loss = loss_func(labels, pred)
    grad = tape.gradient(loss, model.trainable_variables)
    opti.apply_gradients(zip(grad, model.trainable_variables))
    return loss


def draw_picture(datas, loss, wrong_num, w):
    p_samples = [it for it in datas if plane_function(it[0], it[1]) > 0]
    p_samples_x = [x for x, y in p_samples]
    p_samples_y = [y for x, y in p_samples]

    n_samples = [it for it in datas if plane_function(it[0], it[1]) < 0]
    n_samples_x = [x for x, y in n_samples]
    n_samples_y = [y for x, y in n_samples]
    plt.scatter(p_samples_x, p_samples_y, c="red")
    plt.scatter(n_samples_x, n_samples_y, c="blue")
    w1 = w[0][0]
    w2 = w[1][0]
    x = np.linspace(-100, 100)
    y = -1 / w2 * (w1 * x)
    y_true = - 0.5 * x
    plt.plot(x, y, color="b")
    plt.plot(x, y_true, color="r")
    plt.savefig("./" + "perceptron_scatter_fig" + ".png", format='png', bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.show()

    plt.plot(list(range(len(loss))), loss)
    plt.figure()
    plt.plot(list(range(len(wrong_num))), wrong_num)
    plt.show()

if __name__ == "__main__":
    # generate data 
    datas, labels = data_generator(100)
    # define model instances
    perceptron = Perceptron()
    # define optimizer
    opti = tf.keras.optimizers.Adam()
    # optimize the parameters of the model
    wrong_num_list = list()
    loss_list = list()
    for i in range(10000):
        loss = train_step(datas, labels, perceptron)
        if i % 100 == 0:
            num = wrong_num(labels, datas, perceptron)
            loss_list.append(loss)
            wrong_num_list.append(num)
            print("epoch: {}, loss: {}, wrong_num {}".format(i, loss, num))
    w = perceptron.w.numpy()
    print(w)
    draw_picture(datas, loss_list, wrong_num_list, perceptron.w.numpy())


