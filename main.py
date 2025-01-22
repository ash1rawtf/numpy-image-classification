import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

TEST_SAMPLES_COUNT = 1000
EPOCHS = 500
LEARNING_RATE = 0.1


def get_data() -> tuple[NDArray[np.int64]]:
    data = np.array(pd.read_csv("data/train.csv"))
    np.random.shuffle(data)

    data_train = data[TEST_SAMPLES_COUNT:].T
    data_test = data[0:TEST_SAMPLES_COUNT].T

    X_train = data_train[1:]
    X_train = X_train / 255
    Y_train = data_train[0]

    X_test = data_test[1:]
    X_test = X_test / 255
    Y_test = data_test[0]

    return X_train, Y_train, X_test, Y_test


def init_params() -> tuple[NDArray[np.float128]]:
    W1 = (np.random.rand(10, 784) - 0.5).astype("float128")
    b1 = (np.random.rand(10, 1) - 0.5).astype("float128")

    W2 = (np.random.rand(10, 10) - 0.5).astype("float128")
    b2 = (np.random.rand(10, 1) - 0.5).astype("float128")

    return W1, b1, W2, b2


def ReLU(Z) -> NDArray[np.float128]:
    return np.maximum(0, Z)


def ReLU_derivative(Z) -> NDArray[np.float128]:
    return Z > 0


def softmax(Z) -> NDArray[np.float128]:
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)


def forward_prop(X, W1, b1, W2, b2) -> tuple(NDArray[np.float128]):
    Z1 = np.matmul(W1, X) + b1
    A1 = ReLU(Z1)

    Z2 = np.matmul(W2, A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def transform_Y(Y) -> NDArray[np.float128]:
    transformed_Y = np.zeros((Y.size, Y.max() + 1))
    transformed_Y[np.arange(Y.size), Y] = 1
    return transformed_Y.T


def backward_prop(Z1, A1, Z2, A2, W2, X, Y) -> tuple[NDArray[np.float128]]:
    m = Y.size

    dZ2 = A2 - transform_Y(Y)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = np.dot(W2.T, dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2) -> tuple[NDArray[np.float128]]:
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1
    W2 -= LEARNING_RATE * dW2
    b2 -= LEARNING_RATE * db2

    return W1, b1, W2, b2


def get_accuracy(A2, Y) -> np.float64:
    return np.sum(np.argmax(A2, 0) == Y) / Y.size


def training(X, Y) -> tuple[NDArray[np.float128]]:
    W1, b1, W2, b2 = init_params()

    for epoch in range(EPOCHS):
        Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2)

        print(f"Epoch: {epoch} | Accuracy: {get_accuracy(A2, Y)}")

    return W1, b1, W2, b2


def testing(X, Y, W1, b1, W2, b2) -> None:
    Z1, A1, Z2, A2 = forward_prop(X, W1, b1, W2, b2)
    print(f"Testing result accuracy: {get_accuracy(A2, Y)}")


def plot_prediction(index, X, Y, W1, b1, W2, b2) -> None:
    selected_item = X[:, index, None]
    Z1, A1, Z2, A2 = forward_prop(selected_item, W1, b1, W2, b2)
    prediction = np.argmax(A2, 0)

    print(f"Index: {index} | Prediction: {prediction} | Label: {Y[index]}")

    plt.gray()
    plt.imshow(selected_item.reshape((28, 28)) * 255, interpolation="nearest")
    plt.show()


def main() -> None:
    X_train, Y_train, X_test, Y_test = get_data()

    W1, b1, W2, b2 = training(X_train, Y_train)
    testing(X_test, Y_test, W1, b1, W2, b2)

    plot_prediction(0, X_test, Y_test, W1, b1, W2, b2)
    plot_prediction(100, X_test, Y_test, W1, b1, W2, b2)
    plot_prediction(666, X_test, Y_test, W1, b1, W2, b2)
    plot_prediction(900, X_test, Y_test, W1, b1, W2, b2)
    plot_prediction(999, X_test, Y_test, W1, b1, W2, b2)


if __name__ == "__main__":
    main()
