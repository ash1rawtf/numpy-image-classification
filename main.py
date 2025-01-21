import pandas as pd
import numpy as np
from numpy.typing import NDArray

TEST_SAMPLES_COUNT = 1000


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


def main() -> None:
    X_train, Y_train, X_test, Y_test = get_data()
    print(Y_train.dtype)


if __name__ == "__main__":
    main()
