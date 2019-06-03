import csv
import numpy as np
from sklearn.model_selection import train_test_split
import os


def get_labels_wind(X, max_value):

    means = np.mean(X, axis=1)
    labels = np.zeros((X.shape[0], 1))

    # Category for mu(x) > 6 MW
    index = means >= 6.0 / max_value
    labels[index] = 4

    index = means < 6.0 / max_value
    labels[index] = 3

    index = means < 3.0 / max_value
    labels[index] = 2

    index = means < 1.5 / max_value
    labels[index] = 1

    index = means < 0.5 / max_value
    labels[index] = 0

    return labels

def load_wind(filename_X, sample_size=365, verbose=True):

    # Number of 5-min samples in sample_size (days)
    num_samples = 12 * 24 * sample_size

    # Create a 2D ndarray from the .csv file
    with open(filename_X, 'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)

    # Split history of measurements into samples of size sample_size days.
    X = np.empty(shape=(0, num_samples))
    for row_idx, row in enumerate(rows):
        if verbose:
            print(f'Location number: {row_idx + 1}')
        idx = 0
        while (idx + num_samples) < row.size:
            X = np.vstack((X, row[idx: idx + num_samples]))
            idx += num_samples

    # Split data into training data (80%) and testing data (20%).
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=10)

    # Get labels for training data
    Y_train = get_labels_wind(X_train, 16)
    Y_test = get_labels_wind(X_test, 16)

    # Print information about dataset
    if verbose:
        print('Wind data loaded')
        print(f'Total number of samples: {X_train.shape[0] + X_test.shape[0]}')
        print(f'Shape of training dataset: {X_train.shape}')
        print(f'Shape of test datasaet: {X_test.shape}')

    return X_train, Y_train, X_test, Y_test

def get_labels_solar(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)
    print('Shape of labels: ', labels.shape)

    trY = np.tile(labels, (32,1))

    return trY

def load_solar(filename_X, sample_size=2, testsize=0.2, verbose=True):

    with open(filename_X, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)

    X = np.reshape(rows.T,(-1, sample_size * 12 * 24))

    # Normalize data in the [0, 1] range.
    m = np.ndarray.max(rows)
    print("maximum value of solar power", m)
    X = X / m

    # Split data into training data (80%) and testing data (20%).
    X_train, X_test = train_test_split(X, test_size=testsize, random_state=10)

    # Get labels for

    # Print information about dataset
    if verbose:
        print('Solar data loaded')
        print(f'Total number of samples: {X_train.shape[0] + X_test.shape[0]}')
        print(f'Shape of training dataset: {X_train.shape}')
        print(f'Shape of test datasaet: {X_test.shape}')

    return X_train, X_test

def load_solar_2006(filename, file_labels, sample_size=2,
                    testsize=0.2, verbose=True):

    # Load training examples
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    rows = np.array(rows, dtype=float)
    X = np.reshape(rows, (-1, sample_size * 12 * 24))

    # Load labels (corresponding to month)
    with open(file_labels, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    labels = np.array(rows, dtype=int)

    # Same labels for each location
    Y = np.tile(labels, (32,1))

    # Drop extra examples to have consistent shapes
    X = X[:Y.shape[0], :]

    # Split data into training data (80%) and testing data (20%).
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize,
                                                        random_state=10)

    # Print information about dataset
    if verbose:
        print('Solar data loaded')
        print(f'Total number of samples: {X_train.shape[0] + X_test.shape[0]}')
        print(f'Shape of training dataset: {X_train.shape}')
        print(f'Shape of test datasaet: {X_test.shape}')

    return X_train, X_test, y_train, y_test


def load_gan(folder_path):
    pass

if __name__ == '__main__':
    pass
    #X_train, X_test = load_wind('data/less_windy.csv', 5)
