#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

class HMM:
    def __init__(self, A, B, pi, num_observed_value, num_hidden_value):
        self.A = A
        self.B = B
        self.pi = pi

        # number of observed value
        self.num_observed_value = num_observed_value
        # number of hidden value
        self.num_hidden_value = num_hidden_value

        # alpha, forward algorithm
        self.alpha = -1

        # beta, backward algorithm
        self.beta = -1
        

    # sample value from the given distribution
    def sample_from_distribute(self, dist):
        sample = np.random.choice(a=len(dist), size=1, p=dist)
        return sample[0]

    # Based on the arguments to generate data
    def generate_data(self, num):
        # Based on the init distribution select one z
        z = self.sample_from_distribute(self.pi)
        x = self.sample_from_distribute(self.B[z])
        z_list = list()
        x_list = list()
        z_list.append(z)
        x_list.append(x)

        for _ in range(num - 1):
            z = self.sample_from_distribute(self.A[z])
            x = self.sample_from_distribute(self.B[z])
            z_list.append(z)
            x_list.append(x)
        return z_list, x_list

    # calculate probability based on the specified parameters
    def calulate_prob_forward(self, data):
        time_len = len(data)
        alpha = np.zeros(shape=(time_len, self.num_hidden_value))

        # t = 0
        for j in range(self.num_hidden_value):
            alpha[0][j] = self.pi[j] * self.B[j][data[0]]
        
        # t: 1-T
        for t in range(1, time_len):
            for j in range(self.num_hidden_value):
                for i in range(self.num_hidden_value):
                    alpha[t][j] += alpha[t-1][i] * self.A[i][j] * self.B[j][data[t]]
        result = 0
        for j in range(self.num_hidden_value):
            result += alpha[time_len - 1][j]
        return result, alpha
    
    # calculate probability based on the specified parameters
    def calulate_prob_backward(self, data):
        time_len = len(data)
        beta = np.zeros(shape=(time_len, self.num_hidden_value))

        # t = T
        for j in range(self.num_hidden_value):
            beta[-1][j] = 1

        # t = 0
        for t in range(time_len - 2, -1, -1):
            for j in range(self.num_hidden_value):
                for i in range(self.num_hidden_value):
                    beta[t][j] += self.B[i][data[t+1]] * beta[t+1][i] * self.A[j][i]

        result = 0
        for i in range(self.num_hidden_value):
            result += self.B[i][data[0]] * beta[0][i] * self.pi[i]
        return result, beta
    
    def __get_a_ij(self, i, j, alpha, beta, data, e=1e-5):
        numerator = 0
        denominator = 0 + e

        for t in range(1, len(data)):
            numerator += alpha[t-1][i] * beta[t][j] * self.A[i][j] * self.B[j][data[t]]
            denominator += alpha[t-1][i] * beta[t - 1][i]

        return numerator / denominator

    def __get_b_ij(self, i, j, alpha, beta, data, e=1e-5):
        numerator = 0
        denominator = 0 + e
        for t in range(len(data)):
            if data[t] == j:
                numerator += alpha[t][i] * beta[t][i]
            denominator += alpha[t][i] * beta[t][i]

        return numerator / denominator

    def baum_welch_one_step(self, data, e=0.001):
        _, alpha = self.calulate_prob_forward(data)
        
        prob_observed, beta = self.calulate_prob_backward(data)
        
        # pi
        for i in range(self.num_hidden_value):
            self.pi[i] = alpha[1][i] * beta[1][i] / (prob_observed + e)

        # A
        for i in range(self.num_hidden_value):
            for j in range(self.num_hidden_value):
                self.A[i][j] = self.__get_a_ij(i, j, alpha, beta, data, e)

        # B
        for i in range(self.num_hidden_value):
            for j in range(self.num_observed_value):
                self.B[i][j] = self.__get_b_ij(i, j, alpha, beta, data, e)
    
    def decoding(self, data):
        prob_max_matrix = np.zeros(shape=(len(data), self.num_hidden_value))
        id_max_matrix = np.zeros(shape=(len(data), self.num_hidden_value), dtype=np.int32)
        
        prob_max_matrix[0] = self.pi * self.B[:, data[0]]
        id_max_matrix[0] = list(range(self.num_hidden_value))
        for t in range(1, len(data)):
            for i in range(self.num_hidden_value):
                p_max = -1
                id_max = -1
                for j in range(self.num_hidden_value):
                    now = prob_max_matrix[t-1][j] * self.A[j][i] * self.B[i][data[t]]
                    if now > p_max:
                        p_max = now
                        id_max = j
                prob_max_matrix[t][i] = p_max
                id_max_matrix[t][i] = id_max
        
        hidden_list, prob = self.get_hidden_list(prob_max_matrix, id_max_matrix)
        return hidden_list, prob

    def force_decoding(self, data):
        max_prob = -1
        ans = -1
        for hidden_list in itertools.product(list(range(self.num_hidden_value)), repeat=len(data)):
            prob = self.hidden_condition_observed(data, hidden_list)
            if prob > max_prob:
                max_prob = prob
                ans = hidden_list
        return prob, ans

    def get_hidden_list(self,  prob_max_matrix, id_max_matrix):
        max_end_v = -1
        max_end_id = -1
        hidden_list = list()
        for i in range(self.num_hidden_value):
            if prob_max_matrix[-1][i] > max_end_v:
                max_end_v = prob_max_matrix[-1][i]
                max_end_id = i
        hidden_list.append(max_end_id)
        last_id = max_end_id
        for t in range(len(id_max_matrix) - 1, 0, -1):
            last_id = id_max_matrix[t][last_id]
            hidden_list.append(last_id)
        list.reverse(hidden_list)
        return hidden_list, max_end_v
    
    def complete_data_prob(self, observed_list, hidden_list):
        prob = self.pi[hidden_list[0]]
        prob *= self.B[hidden_list[0]][observed_list[0]]

        for t in range(1, len(hidden_list)):
            prob *= self.A[hidden_list[t-1]][hidden_list[t]]
            prob *= self.B[hidden_list[t]][observed_list[t]]
        return prob

    def hidden_prob(self, hidden_list):
        prob = self.pi[hidden_list[0]]
        for t in range(1, len(hidden_list)):
            prob *= self.A[hidden_list[t-1]][hidden_list[t]]
        return prob

    def hidden_condition_observed(self, observed_list, hidden_list):
        observed_prob, _ = self.calulate_prob_forward(observed_list)
        complete_data_prob = self.complete_data_prob(observed_list, hidden_list)
        return complete_data_prob / observed_prob


def plt_figure(data):
    fig, ax = plt.subplots()
    ax.plot(list(range(len(prob_list))), prob_list)
    ax.set_xlabel("epoch")
    ax.set_ylabel("prob")
    plt.savefig("./" + "optimize" + ".png", format='png', bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.show()

if __name__ == "__main__":
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    A = np.array([
        [0, 1, 0, 0],
        [0.4, 0, 0.6, 0],
        [0, 0.4, 0, 0.6],
        [0, 0, 0.5, 0.5]])
    B = np.array([
        [0.5, 0.5],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.8, 0.2]])
    hmm = HMM(A, B, pi, 2, 4)

    print("===============generate data==============")
    hidden_list, observed_list = hmm.generate_data(5)
    print("hidden list: ", hidden_list)
    print("observed list: ", observed_list)
    print("===============forword algorithm==============")
    prob, _ = hmm.calulate_prob_forward(observed_list)
    print("forward prob: ", prob)
    prob, _ = hmm.calulate_prob_backward(observed_list)
    print("backward prob: ", prob)

    print("===============decoding==============")
    decoding_hidden, prob = hmm.decoding(observed_list)
    print("decoding hidden: ", decoding_hidden)
    print("complete prob: ", hmm.complete_data_prob(observed_list, decoding_hidden))
    print("condition prob: ", hmm.hidden_condition_observed(observed_list, decoding_hidden))
    prob, ans = hmm.force_decoding(observed_list)
    print("force decoding hidden list", ans)
    print("force decoding prob", prob)
    print("===============optimizing algorithm==============")
    # if the sequence is too long, the fiting effect is not good
    prob_list = list()
    for i in range(50):
        prob, _ = hmm.calulate_prob_forward(data=observed_list)
        print("epoch: ", i, "probability: ", prob)
        hmm.baum_welch_one_step(observed_list, e=0.0001)
        prob_list.append(prob)
    
    plt_figure(prob_list)
    
