import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import os

#  三类    A              B           C
#  比重   1/6            2/6         1/2
#  均值  (0, 0)         (2, 5)      (5, 5)
#  方差    
#        0.5, 0          0.4,0       0.4,0
#        0, 0.8          0,0.3       0,0.3

def data_generator():
    # label A
    mu_A = np.array([0, 0], dtype=np.float32)
    cov_A = np.array([[0.5, 0], [0, 0.8]], dtype=np.float32)
    num_A = 100
    data_A = np.random.multivariate_normal(mean=mu_A, cov=cov_A, size=num_A)
    label_A = np.empty(num_A)
    label_A.fill(0)
    
    # label B
    mu_B = np.array([2, 5], dtype=np.float32)
    cov_B = np.array([[0.4, 0], [0, 0.3]], dtype=np.float32)
    num_B = 200
    data_B = np.random.multivariate_normal(mean=mu_B, cov=cov_B, size=num_B)
    label_B = np.empty(num_B)
    label_B.fill(1)
    
    # label C
    mu_C = np.array([5, 5], dtype=np.float32)
    cov_C = np.array([[0.4, 0], [0, 0.3]], dtype=np.float32)
    num_C = 300
    data_C = np.random.multivariate_normal(mean=mu_C, cov=cov_C, size=num_C)
    label_C = np.empty(num_C)
    label_C.fill(2)

    data = np.concatenate((data_A, data_B, data_C), axis=0)
    label = np.concatenate((label_A, label_B, label_C), axis=0)
    return data, label

def plt_data(data):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], s=5)
    plt.show()

class GMM():
    def __init__(self):
        # the number of class in dataset
        self.classes_num = -1
        # the dimension of data
        self.data_dim = -1
        # the number of data
        self.data_num = -1
        # the mean of every distribution.
        self.mu = -1
        # the convariance of every distribution
        self.cov = -1
        # mixing coefficient
        self.p = -1
    
    def __set_init_para(self, data, label):
        unique_label = np.unique(label)
        self.classes_num = len(unique_label)
        self.data_dim = len(data[0])
        self.data_num = data.shape[0]

        # mixing coefficient
        self.p = np.ones(self.classes_num) / self.classes_num
        # mu
        self.mu = np.array([
        [1, -1],
        [1, 4],
        [4.8, 5]
        ])
        # cov
        data_cov = np.cov(m=data.T)
        data_cov = np.expand_dims(data_cov, axis=0)
        self.cov = np.repeat(a=data_cov, repeats=self.classes_num, axis=0)


    def fit(self, data, label, epoch_num):
        self.__set_init_para(data, label)
        for epoch in range(epoch_num):
            zp = self.E_step(data)
            self.M_step(data, zp)
            print("acc: ", self.metrics(label, zp))
    # update z
    def E_step(self, data):
        zp = np.zeros(shape=(self.data_num, self.classes_num))
        for i in range(self.data_num):
            # denominator
            z_list = np.zeros(shape=(self.classes_num))
            for j in range(self.classes_num):
                z_list[j] = self.p[j] * multivariate_normal.pdf(data[i], self.mu[j], self.cov[j])
            zp[i] = z_list / np.sum(z_list)
        return zp

    # update mu, sigma 
    def M_step(self, data, zp):
        # update p
        self.p = np.sum(zp, axis=0) / self.data_num
        for i in range(self.classes_num):
            numerator = np.zeros(shape=self.data_dim)
            denominator = np.sum(zp, axis=0)[i]
            for j in range(self.data_num):
                numerator += data[j] * zp[j][i]
            # update mu
            self.mu[i] = numerator / denominator
            # initial numerator
            numerator = np.zeros(shape=(self.data_dim, self.data_dim))

            for j in range(data_num):
                v = np.expand_dims(data[j] - self.mu[i], axis=0)
                numerator += np.dot(v.T, v) * zp[j][i]
            # update sigma
            self.cov[i] = numerator / denominator
    
    def metrics(self, train_y, zp):
        pred_y = np.argmax(zp, axis=-1)
        diff_indicator = np.equal(pred_y, train_y).astype(dtype=np.int32)
        return np.sum(diff_indicator) / len(pred_y)

if __name__ == "__main__":
    data, label = data_generator()
    plt_data(data)
    data_num = data.shape[0]
    gmm = GMM()
    gmm.fit(data, label, epoch_num=15)


