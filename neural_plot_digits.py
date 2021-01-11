import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_random_gray_scale_digit():
    gray_scale_data = pd.read_csv('../ml_python/sample_data/MNIST.csv', header=None).to_numpy()
    row, col = gray_scale_data.shape

    digit_gray_scale = gray_scale_data[:, :col - 1]
    digit_label = gray_scale_data[:, col - 1]

    digit_label[digit_label == 10] = 0

    fig1, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            x = np.random.randint(row)
            digit = digit_gray_scale[x].reshape((20, 20))
            axs[i, j].imshow(digit, cmap='gray')
            axs[i, j].get_xaxis().set_ticks([])
            axs[i, j].get_yaxis().set_ticks([])
            axs[i, j].set_xlabel(int(digit_label[x]))
    fig1.suptitle('Examples of handwritten digits in gray scale')


if __name__ == '__main__':
    draw_random_gray_scale_digit()