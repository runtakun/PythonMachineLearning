
from perceptron import Perceptron
import numpy as np


def main():
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    x = df.iloc[0:100, [0, 2]].values

    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(x, y)

    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')

    plt.show()

if __name__ == '__main__':
    main()
