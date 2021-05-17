import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math


class Network(object):
    def __init__(self, sizes, nigains=None):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""

        # Number of neurons for each layer
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.gains = []

        # If none all nigains 1.0
        if nigains != None:
            self.gains = nigains
        else:
            for _ in range(self.num_layers):
                self.gains.append(1.0)
        # Init the biases and weights
        self.biases = [np.full((y, 1), 0.0, dtype=float) for y in sizes[1:]]
        self.weights = [np.random.normal(loc=0.0, scale=0.1, size=(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

        self.errors = []
        self.cost_ = []
        self.ni = []  # list to store all the net_input vectors, layer by layer
        self.activations = []  # list to store all the activations, layer by layer

    def setupTrainingData(self, Xt, y):
        yt = np.zeros((len(y), self.sizes[-1]))
        for n, val in enumerate(y.astype(int)):
            yt[n, val] = 1.0
        training_data = np.append(Xt, yt, axis=1)
        return training_data

    def getTrainingDataForOneImage(self, training_data, imgNumber=0):
        X = training_data[:, :self.sizes[0]]
        Y = training_data[:, -self.sizes[-1]:]
        return X[imgNumber], Y[imgNumber]

    def learnOneImage(self, x , y, eta):
        y.resize(len(y), 1)
        x.resize(len(x), 1)

        # feedforward
        ypred = self.predict(x)

        # backward pass
        # if error < 0 the weights must increase
        dcost = error = self.cost_derivative(ypred, y)  # Derivative cost function
        self.errors.append(error)
        dactv = self.sigmoidDerivate(self.activations[-1])  # Derivative Activation Function

        gradient = dcost * dactv
        self.biases[-1] -= eta * gradient
        self.weights[-1] -= eta * np.dot(gradient, self.activations[-2].T)

        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.

        pcost = dcost.T  # previous cost
        # range(start, stop, step)
        for l in range(2, self.num_layers):
            herror = np.dot(pcost, self.weights[-l + 1])
            hcost = np.sum(herror, axis=0)
            dactv = self.sigmoidDerivate(self.activations[-l].T)
            gradient = hcost * dactv
            pcost = hcost

            self.biases[-l] -= eta * gradient.T
            self.weights[-l] -= eta * np.dot(gradient.T, self.activations[-l - 1].T)

        actvError = np.dot(pcost, self.weights[-l])
        return actvError

    def learnOneEpoch(self, training_data, eta):
        X = training_data[:, :self.sizes[0]]
        Y = training_data[:, -self.sizes[-1]:]

        # Iterate through all images given in X
        for n, x in enumerate(X):
            y = Y[n]
            self.learnOneImage(x, y, eta)

    def learn(self, training_data, epochs, eta, cost_min=0.1):
        self.errors.clear()
        self.cost_.clear()
        for e in range(epochs):
            self.learnOneEpoch(training_data, eta)

            # Calculate the cost value in %
            cost = self.PE_Cost()
            print("Cost in %d epoch = %2.2f " % (e, cost) + str("%"))

            # Stop learning if cost < cost_min [%]
            if cost < cost_min:
                break

    def predict(self, a, type=0):
        """ l = self.num_layers, range from first hidden layer until output layer included """
        self.ni.clear()  # list to store all the net_input vectors, layer by layer
        self.activations.clear()  # list to store all the activations, layer by layer

        self.activations.append(a)
        for l, (b, w) in enumerate(zip(self.biases, self.weights)):
            h = np.dot(w * self.gains[l], a)
            self.ni.append(h)
            a = self.sigmoid(h + b)
            self.activations.append(a)

        actv = self.activations[-1]

        if type == 0:
            if self.sizes[-1] <= 2:
                ypred = np.where(actv >= 0.5, 1, 0)
                return ypred
            else:
                sm = self.softmax(actv)
                am = np.argmax(sm, axis=0)
                ypred = np.zeros_like(actv)
                ypred[am] = 1.0
                return ypred
        else:
            if self.sizes[-1] == 1:
                ypred = np.where(a >= 0.5, 1, 0)
                return ypred.T
            elif self.sizes[-1] == 2:
                ypred = np.where(a >= 0.5, 1, 0)
                return ypred[0]
            else:
                sm = self.softmax(actv)
                am = np.argmax(sm, axis=0)
                ypred = np.array([am])
                ypred.resize(len(am), 1)
                return ypred

    def PE_Cost(self):
        # Percent errors
        cost = (np.sum([(e ** 2) / 2 for e in self.errors]) * 100)/len(self.errors)
#        print("Epoch %d an cost %2.2f " % (epoch, cost) + str('%'))

        # Put all cost values in a list to display them later
        self.cost_.append(cost)
        self.errors.clear()
        return cost

    def sigmoid(self, net_input):
        """The sigmoid function."""
        sigm = 1.0/(1.0 + np.exp(-np.clip(net_input, -250, 250)))
        return sigm

    def sigmoidDerivate(self, output_activation):
        """Derivative of the sigmoid function."""
        derivative = output_activation * (1 - output_activation)
        return derivative

    def cost_derivative(self, output_activation, label):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        cost_derivative = output_activation - label
        return cost_derivative

    def softmax(self, numbers):
        sm = np.exp(numbers)
        return sm

    def CalculatePredictionPercentage(self, X, y):
        p = self.predict(X.T, type=1)
        if self.sizes[-1] == 1:
            ys = np.subtract(y.T, p.T)
            sumwrong = (ys != 0).sum()
            percentage = (sumwrong * 100)/ys.size
            return percentage
        elif self.sizes[-1] == 2:
            ys = np.subtract(y.T[0], p.T)
            sumwrong = (ys != 0).sum()
            percentage = (sumwrong * 100)/ys.size
            return percentage
        else:
            ys = np.subtract(y, p)
            sumwrong = (ys != 0).sum()
            percentage = (sumwrong * 100)/ys.size
            return percentage

    def saveNetInputsAndActivations(self):
        # For each input during learning are net_inputs and activations updated
        np.savez_compressed('netinputs.npz', *self.ni)
        np.savez_compressed('activations.npz', *self.activations)

    def getNetInputsAndActivations(self):
        # For each input during learning are net_inputs and activations updated
        net_inputs = []
        nis = np.load('netinputs.npz')
        for f in nis.files:
            net_inputs.append(nis[f])

        activations = []
        actv = np.load('activations.npz')
        for f in actv.files:
            activations.append(actv[f])

        return net_inputs, activations

    def saveBiasesAndWeights(self, prefix=None):
        path = "./BWSaved/"
        bFilename = "biases.npz"
        wFilename = "weights.npz"
        if prefix is None:
            biasesFilename = bFilename
            weightsFilename = wFilename
        else:
            biasesFilename = prefix + bFilename
            weightsFilename = prefix + wFilename

        np.savez_compressed(path + biasesFilename, *self.biases)
        np.savez_compressed(path + weightsFilename, *self.weights)

    def getBiasesAndWeights(self, prefix=None):
        path = "./BWSaved/"
        bFilename = "biases.npz"
        wFilename = "weights.npz"
        if prefix is None:
            biasesFilename = bFilename
            weightsFilename = wFilename
        else:
            biasesFilename = prefix + bFilename
            weightsFilename = prefix +wFilename

        biases = np.load(path + biasesFilename)
        for n, f in enumerate(biases.files):
            self.biases[n] = biases[f]

        weights = np.load(path + weightsFilename)
        for n, f in enumerate(weights.files):
            self.weights[n] = weights[f]

    def plotCost(self):
        # plot weights updates for each epoch
        plt.plot(range(1, len(self.cost_) + 1), self.cost_, marker='o', markersize=3)
        plt.title("NN1: Gradient descend weights updates")
        plt.xlabel('Epochs')
        plt.ylabel('Wrong predictions in ' + str('%'))
        plt.show()

    def DisplayVectors(self, X, Y, ypredicts ,n, nepoches, firstlast=True):
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        markers = ('s', 'x', 'o', '^', 'v')
        M = X
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        rows, cols = M.shape
        for r in range(0, rows):
            #        x1    x2      y1    y2
            ax.plot(M[r, 0], M[r, 1], 'k', marker=markers[np.argmax(Y[r])])
            ax.plot([M[r, 0], 3], [M[r, 1], np.argmax(ypredicts[r])], colors[np.argmax(ypredicts[r])], linestyle='dashed')
            ax.plot(3, np.argmax(Y[r]), marker='o', color=colors[np.argmax(Y[r])])

        ax.axis('equal')
        ax.grid(b=True, which='major')
        # Plot only first and last figure
        if n != 0 and n < (nepoches - 1) and firstlast:
            plt.close(fig)
        else:
            plt.title("Predictions after %d epoch" % (n+1))
            plt.show()


def plot_data(X, y, titletxt=""):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
#                    c=colors[idx],
                    c='k',
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.xlabel('x1 [cm]')
    plt.ylabel('x2 [cm]')
    plt.title(titletxt)
    plt.legend(loc='upper right')
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.01):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))
    I = np.array([xx1.ravel(), xx2.ravel()]).T

    # Do the predictions
    Z = classifier.predict(I.T, type=1)

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    # x = (x,2)  with x number of samples
    # y = (1,x)
    y = y.T
    y = y[0].astype(int)
    for idx, cl in enumerate(np.unique(y)):
        xv = X[y == cl, 0]
        yv = X[y == cl, 1]
        plt.scatter(x=xv,
                    y=yv,
                    alpha=0.8,
#                    c=colors[idx],
                    c='k',
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


def plot_weights(net, Dim2=False, Dim3=False):
    '''
    Plot all weights of NN  in 2D and 3D

    3D Plots:
    https://chris35wills.github.io/courses/PythonPackages_matplotlib/matplotlib_3d/
    '''

    net.getBiasesAndWeights()
    for l, w in enumerate(net.weights):
        if Dim2 is True:
            # Plot weights[0] in 2D
            w = net.weights[l]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.matshow(w, cmap='afmhot')
            if l == 0:
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Input nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden nodes (%d)" % w.shape[0])
            elif l < (net.num_layers - 2):
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Hidden input nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden output nodes (%d)" % w.shape[0])
            else:
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Hidden nodes (%d)" % w.shape[1])
                ax.set_ylabel("Output nodes (%d)" % w.shape[0])

            plt.tight_layout()
            plt.show()

        if Dim3 is True:
            # Plot weights[0] in 3D
            ny, nx = w.shape
            x = np.linspace(0, nx, nx, endpoint=False)
            y = np.linspace(0, ny, ny, endpoint=False)
            xv, yv = np.meshgrid(x, y)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            dem3d = ax.plot_surface(xv, yv, w, linewidth=0, rstride=1, cstride=1, cmap='afmhot')
            if l == 0:
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Input nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden nodes (%d)" % w.shape[0])
            elif l < (net.num_layers - 2):
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Hidden input nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden output nodes (%d)" % w.shape[0])
            else:
                ax.set_title('Hidden weights\n\n')
                ax.set_xlabel("Hidden nodes (%d)" % w.shape[1])
                ax.set_ylabel("Output nodes (%d)" % w.shape[0])
            plt.colorbar(dem3d, shrink=0.4, aspect=10)
            plt.tight_layout()
            plt.show()


def DisplayPredict(net, I):
    I.resize(I.size, 1)
    ypred = net.predict(I)
    y_predict = np.argmax(ypred)
    net.saveNetInputsAndActivations()

    # Plot net_inputs and activations of NN
    net_inputs, activations = net.getNetInputsAndActivations()
    net.getBiasesAndWeights()

    for n, actv in enumerate(activations):
        if n < (net.num_layers - 1):
            # Plot activations[0] in 2D
            actv = activations[n]
            imagesize = int(math.sqrt(actv.size))
            mod = imagesize % 2
            if mod == 0:
                actv.resize(imagesize, imagesize)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dem2d = ax.matshow(actv, cmap='afmhot')
            if n == 0:
                ax.set_title('Activation input layer\n\n')
            else:
                ax.set_title('Activation hidden layer %d\n\n' % n)
            plt.colorbar(dem2d, shrink=0.4, aspect=10)
            plt.tight_layout()
            plt.show()

            # Plot weights[0] in 2D
            w = net.weights[n]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.matshow(w, cmap='afmhot')
            if n == 0:
                ax.set_title('Weights hidden layer %d\n\n' % (n+1))
                ax.set_xlabel("Input nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden nodes (%d)" % w.shape[0])
            elif n < (net.num_layers - 2):
                ax.set_title('Weights hidden layer %d\n\n' % (n+1))
                ax.set_xlabel("Hidden nodes (%d)" % w.shape[1])
                ax.set_ylabel("Hidden nodes (%d)" % w.shape[0])
            else:
                ax.set_title('Weights output layer\n\n')
                ax.set_xlabel("Hidden nodes (%d)" % w.shape[1])
                ax.set_ylabel("Output nodes (%d)" % w.shape[0])
            plt.tight_layout()
            plt.show()

            # Plot net_inputs[0] in 2D
            ni = net_inputs[n]
            imagesize = int(math.sqrt(ni.size))
            mod = imagesize % 2
            if mod == 0:
                ni.resize(imagesize, imagesize)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dem2d = ax.matshow(ni, cmap='afmhot')
            if n < (net.num_layers - 2):
                ax.set_title('Net_input hidden layer %d\n\n' % (n + 1))
            else:
                ax.set_title('Net_input output layer\n\n')
            plt.colorbar(dem2d, shrink=0.4, aspect=10)
            plt.tight_layout()
            plt.show()
        else:
            # Plot activations[2] in 2D
            actv = activations[n]
            actv.resize(actv.size, 1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dem2d = ax.matshow(actv, cmap='afmhot')
            ax.set_title('Activation output layer\n\n')
            plt.colorbar(dem2d, shrink=0.4, aspect=10)
            plt.tight_layout()
            plt.show()

    return y_predict
