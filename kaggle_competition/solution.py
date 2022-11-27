import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def softmax(x):
    return np.exp(log_prob(x))


def logsumexp(x):
    max_ = x.max(1)
    return max_ + np.log(np.exp(x - max_[:, np.newaxis]).sum(1))


def log_prob(x):
    return x - logsumexp(x)[:, np.newaxis]


class RegressionLogistique(object):

    def __init__(self, n_inputs, n_classes, reg=0.1, n_steps=100):
        self.w = np.random.uniform(-0.05, 0.05, size=(n_inputs, n_classes)).astype(float)
        #self.w[-1] = 1
        self.reg = reg
        self.n_steps = n_steps
        self.n_classes = n_classes

    def preprocess(self, X):
        # normalisation des données
        mu = X[:, :20].mean(axis=0)
        sigma = X[:, :20].std(axis=0)
        X[:, :20] = (X[:, :20] - mu) / sigma
        return X

    def augment_bias(self, X):
        ones = np.ones_like(X[:, 0])[:, np.newaxis]
        # print(ones.shape, X.shape)
        return np.hstack((X, ones))

    def compute_predictions(self, test_data):
        return np.dot(test_data, self.w).argmax(1) + 1
        # return np.dot(self.augment_bias(test_data), self.w).argmax(1) + 1

    def test(self, X, y):
        """Retourne le taux d'erreur pour un batch X
        """
        err_rate = (self.compute_predictions(X) != y).mean()
        print("error rate", err_rate)
        return err_rate

    def gradient_regularizer(self):
        # added this function because the regularizer is the same everywhere
        penalty = self.reg * self.w
        # no bias
        penalty[-1] = 0
        return penalty

    def loss(self, X, y):
        y_onehot = (np.eye(int(y.max() + 1))[y]).astype(int)
        # Z = - X @ self.w
        # N = X.shape[0]
        # loss = 1 / N * (np.trace(X @ self.w @ y_onehot.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        # return loss
        # m = y_onehot.shape[0]
        # p = softmax(X)
        # log_likelihood = -np.log(p[range(m), y_onehot])
        # loss = np.sum(log_likelihood) / m
        return 0

    def gradient(self, X, y_onehot, mu=0.01):
        #y_onehot = np.eye(int(y.max() + 1))[y]
        Z = - X @ self.w
        P = softmax(Z)
        print("pre bias", X.shape)
        # X = self.augment_bias(X)
        print("post bias", X.shape)
        sm = softmax(np.dot(X, self.w))
        # fill grad!
        print(y_onehot.shape)
        grad = 1/(X.shape[0]) * (X.T @ (y_onehot - P)) + 2 * mu * self.w
        return grad + self.gradient_regularizer()

    def descente_gradient(self, X, y, max_iteration=3, eta=0.1, mu=0.01):
        # y_onehot = np.zeros((y.size, int(y.max()+1)))
        # y_onehot[np.arange(y.size), y] = 1
        y_onehot = np.eye(int(y.max()+1))[y]
        print("fd",y_onehot.shape)
        for i in range(max_iteration):
            self.w -= eta * self.gradient(X, y_onehot)

        return self.w

    def train(self, X, y, stepsize=0.5):
        losses = []
        errors = []

        for n in range(self.n_steps):
            print("step:", n)
            # grad = self.gradient(X, y)
            # print("grad", grad)
            # coef = self.coefficients_sgd(X)
            # self.w -= stepsize * grad
            self.w = self.descente_gradient(X, y)
            loss = self.loss(X, y)
            error = self.test(X, y)
            # Update losses
            losses.append(loss)

            # Update errors
            errors.append(error)
        #print("Training completed: the train error is {:.2f}%".format(errors[-1]*100))
        return np.array(losses), np.array(errors)

    def predict(self, X_test):
        Z = - X_test @ self.w
        P = softmax(Z)  # TODO
        return np.argmax(P, axis=1)

    # Make a prediction with coefficients
    def predict2(self, row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row) - 1):
            yhat += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + np.exp(-yhat))

    # Estimate logistic regression coefficients using stochastic gradient descent
    def coefficients_sgd(self, X, l_rate=0.3, n_epoch=10):
        coef = [0.0 for i in range(len(X[0]))]
        for epoch in range(n_epoch):
            sum_error = 0
            for row in X:
                yhat = self.predict2(row, coef)
                error = row[-1] - yhat
                sum_error += error ** 2
                coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
                for i in range(len(row) - 1):
                    coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        return min(coef)


class Perceptron(object):

    # Split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds = cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Make a prediction with weights
    def predict(self, row, weights):
        activation = weights[0]
        for i in range(len(row) - 1):
            activation += weights[i + 1] * row[i]
        return 1.0 if activation >= 0.0 else 0.0

    # Estimate Perceptron weights using stochastic gradient descent
    def train_weights(self, train, l_rate=0.1, n_epoch=5):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = predict(row, weights)
                error = row[-1] - prediction
                sum_error += error ** 2
                weights[0] = weights[0] + l_rate * error
                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        return weights

    # Perceptron Algorithm With Stochastic Gradient Descent
    def perceptron(train, test, l_rate, n_epoch):
        predictions = list()
        weights = train_weights(train, l_rate, n_epoch)
        for row in test:
            prediction = predict(row, weights)
            predictions.append(prediction)
        return (predictions)


if __name__ == '__main__':
    train_set = pd.read_csv('train.csv')
    train_set = train_set.values
    test_set = pd.read_csv('test.csv')
    test_set = test_set.values
    X_train = train_set[:, 1:20]
    y_train = train_set[:, -1].astype(int)
    nbr_classes = np.unique(y_train)

    # sample_verify = pd.read_csv('sample_submission.csv')
    model = RegressionLogistique(len(X_train[0]), len(nbr_classes))
    losses, errors = model.train(X_train, y_train)
    print("losses:", losses)
    print("errors:", errors)
    model.predict(test_set[:, 1:20])

    ###############################################################

    # Test the Perceptron algorithm on the sonar dataset
    # load and prepare data
    train = pd.read_csv('train.csv')
    train = train.values
    # evaluate algorithm
    n_folds = 3
    l_rate = 0.01
    n_epoch = 500
    scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
