import NN as nn
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Run example:
    example = 2

    # NN with 2 neurons in first, 3 neurons in second and 1 neuron in final layer.
    if example == 1:
        X = np.array([[1, 2], [1, 3], [2, 3], [2, 1], [3, 1], [3, 2]], dtype=float)
        y = np.array([[0, 0, 0, 1, 1, 1]], dtype=float).T
        net = nn.Network([2, 3, 1])
    elif example == 2:
        X = np.array([[1, 2], [1, 3], [2, 2], [2, 3], [2.5, 1.5], [2, 1], [3, 1], [3, 2]], dtype=float)
        y = np.asarray([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=float).T
        nigains = [9.0, 5.0, 1.0]
        net = nn.Network([2, 80, 80, 1], nigains)


    # Standardization which gives our data the property of a standard normal distribution,
    # which helps gradient descend learning to converge more quickly. Standardization shifts
    # the mean of each feature so that it is centered at zero and each feature has a standard
    # deviation of 1.
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X_Sigm = X_std

    # Define number Class1 items in X (first ones)
    nn.plot_data(X, y.T[0], titletxt="Features")
    nn.plot_data(X_Sigm, y.T[0], titletxt="Features standardized")

    # Train the network
    # Epoch=30,  learning rate=0.1
    training_data = np.append(X_Sigm, y, axis=1)
    if example == 1:
        net.learn(training_data, 100, 0.1)
    elif example == 2:
        net.learn(training_data, 200, 0.02)

    # plot weights updates for each epoch
    plt.plot(range(1, len(net.cost_) + 1), net.cost_, marker='o', markersize=3)
    plt.title("NN1: Gradient descend weights updates")
    plt.xlabel('Epochs')
    plt.ylabel('Mean squared error (MSE)')
    plt.show()

    # Calculate prediction accuracy
    percentage = net.CalculatePredictionPercentage(X_Sigm, y)
    print("Wrong predictions =  %2.2f " % (percentage,) + str('%'))

    # plot decision regions
    nn.plot_decision_regions(X_Sigm, y, classifier=net)
    plt.xlabel('x1 [cm]')
    plt.ylabel('x2 [cm]')
    plt.legend(loc='upper left')
    plt.show()
