import pandas as pd
import numpy as np
import scipy.special as scsp
import matplotlib.pyplot as plt

from neural_get_train_and_valid import get_train_and_valid
from neural_one_vs_all_training import one_vs_all_training
from logistic_cost_function import logistic_cost_function
from neural_plot_digits import draw_random_gray_scale_digit
from neural_one_vs_all_pred import one_vs_all_predict


'''Load data set'''
gray_scale_data = pd.read_csv('../ml_python/sample_data/MNIST.csv', header=None).to_numpy()
row, col = gray_scale_data.shape


'''Visualize the data'''
draw_random_gray_scale_digit(gray_scale_data)


'''Data partition'''
train_size_per_digit = 400
train_set, valid_set = get_train_and_valid(gray_scale_data, train_size_per_digit)

train_x = train_set[:, :col - 1]
train_y = train_set[:, col - 1]
train_y[train_y == 10] = 0

valid_x = valid_set[:, :col - 1]
valid_y = valid_set[:, col - 1]
valid_y[valid_y == 10] = 10


'''Test for logistic regression cost function'''
test_theta = np.zeros(train_x.shape[1])
test_cost = logistic_cost_function(test_theta, train_x, train_y)
print('test logistic cost function value:', test_cost, '\n')


'''One vs all training'''
all_theta = one_vs_all_training(train_x, train_y, lamb=0.1, num_iter=50)


'''One-layer activation'''
m = train_x.shape[0]
x = np.insert(train_x, 0, np.ones(m), axis=1)
z = x @ all_theta.T
activation = scsp.expit(z)


'''Visualizing one-layer activation'''
fig2, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        x = np.random.randint(m)
        a = activation[x].reshape((5, 2))
        axs[i, j].imshow(a, cmap='gray')
        axs[i, j].get_xaxis().set_ticks([])
        axs[i, j].get_yaxis().set_ticks([])
        axs[i, j].set_xlabel(int(train_y[x]))
fig2.suptitle("Example: One-layer activation")


'''In sample predicting'''
in_sample_pred = one_vs_all_predict(all_theta, train_x)
in_accuracy = np.sum(in_sample_pred == train_y)/len(in_sample_pred)
print('In sample accuracy:', '{percent:.2%}'.format(percent=in_accuracy))


'''Out sample predicting'''
out_sample_pred = one_vs_all_predict(all_theta, valid_x)
out_accuracy = np.sum(out_sample_pred == valid_y)/len(out_sample_pred)
print('Out sample accuracy:', '{percent:.2%}'.format(percent=out_accuracy))


'''output trained theta'''
save_text = input('Do you want to save the trained theta? [y/n]: ',)
if save_text.strip().lower() == 'y':
    np.savetxt('../ml_python/sample_data/one_layer_neural_theta.csv', all_theta, delimiter=',')
