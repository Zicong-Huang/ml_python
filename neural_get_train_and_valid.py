import numpy as np


def get_train_and_valid(data, train_size_per_label, num_label=10):
    row, col = data.shape

    label = data[:, col-1]
    label[label == 10] = 0

    train_size = train_size_per_label * num_label
    valid_size = row - train_size
    valid_size_per_label = valid_size // num_label

    train_set = np.zeros((train_size, col))
    valid_set = np.zeros((valid_size, col))
    for i in range(num_label):
        label_is_i_data = data[label == i]

        start_train = i*train_size_per_label
        end_train = start_train + train_size_per_label

        start_valid = i*valid_size_per_label
        end_valid = start_valid + valid_size_per_label

        train_set[start_train:end_train, :] = label_is_i_data[:train_size_per_label, :]
        valid_set[start_valid:end_valid, :] = label_is_i_data[train_size_per_label:, :]

    return train_set, valid_set
