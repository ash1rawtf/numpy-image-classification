import pandas as pd
import numpy as np
from numpy.typing import NDArray

TEST_SAMPLES_COUNT = 1000
LEARNING_RATE = 0.1


def get_data() -> tuple[NDArray[np.int64]]:
    data = np.array(pd.read_csv("data/train.csv"))
    np.random.shuffle(data)

    data_train = data[TEST_SAMPLES_COUNT:].T
    data_test = data[0:TEST_SAMPLES_COUNT].T

    X_train = data_train[1:]
    Y_train = data_train[0]

    X_test = data_test[1:]
    Y_test = data_test[0]

    return X_train, Y_train, X_test, Y_test


def init_params() -> tuple[NDArray[np.float64]]:
    print(np.random.rand(10, 10) - 0.5)
    W1 = (np.random.rand(10, 784) - 0.5).astype("float128")
    b1 = (np.random.rand(10, 1) - 0.5).astype("float128")

    W2 = (np.random.rand(10, 10) - 0.5).astype("float128")
    b2 = (np.random.rand(10, 1) - 0.5).astype("float128")

    return W1, b1, W2, b2


def ReLU(Z) -> NDArray[np.float64]:
    return np.maximum(0, Z)


def softmax(Z) -> NDArray[np.float64]:
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward_prop(X, W1, b1, W2, b2) -> None:
    Z1 = np.dot(W1, X) + b1
    A1 = ReLU(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    return A2


def main() -> None:
    X_train, Y_train, X_test, Y_test = get_data()
    W1, b1, W2, b2 = init_params()

    A2 = forward_prop(X_train, W1, b1, W2, b2)
    print(A2)


if __name__ == "__main__":
    main()
