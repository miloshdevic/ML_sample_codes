import numpy as np
import math
# import matplotlib.pyplot as plt
# iris = np.genfromtxt('')

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        means = np.zeros(len(iris[0])-1)
        for k in range(0, len(iris[0])-1):
            sum = 0
            for i in range(0, len(iris)):
                sum += iris[i][k]
            means[k] = sum/len(iris)
        return means

    def covariance_matrix(self, iris):
        cov_mat = np.zeros((4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                cov_mat[i][j] = self.covariance_calculator(iris, i, j)
        return cov_mat

    def feature_means_class_1(self, iris):
        means = np.zeros(len(iris[0])-1)
        for k in range(0, len(iris[0])-1):
            sum = 0
            n = 0
            for i in range(0, len(iris)):
                if iris[i][4] == 1:
                    sum += iris[i][k]
                    n +=1
            if n != 0:
                means[k] = sum / n
        return means

    def covariance_matrix_class_1(self, iris):
        cov_mat = np.zeros((4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                if iris[i][4] == 1:
                    cov_mat[i][j] = self.covariance_calculator_class_1(iris, i, j)
        return cov_mat

    def covariance_calculator(self, iris, x, y):
        means = self.feature_means(iris)
        sum = 0
        for i in range(0, len(iris)):
            sum += (iris[i][x]-means[x])*(iris[i][y]-means[y])
        return sum/len(iris)

    def covariance_calculator_class_1(self, iris, x, y):
        means = self.feature_means_class_1(iris)
        sum = 0
        for i in range(0, len(iris)):
            if iris[i][4] == 1:
                sum += (iris[i][x]-means[x])*(iris[i][y]-means[y])
        return sum/len(iris)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def compute_predictions(self, test_data):
        nbr_test = test_data.shape[0]
        classes_pred = np.zeros(nbr_test, dtype=int)
        counts = np.ones((nbr_test, len(self.label_list)))

        # pour chaque point de test
        for (i, ex) in enumerate(test_data):
            kernel = np.zeros(len(self.train_inputs))
            has_neighbours = False

            # calculer l'inverse de la somme des kernels
            for j in range(0, len(self.train_inputs)):
                dist_vect = distance(ex, self.train_inputs[j])
                if dist_vect < self.h:
                    kernel[j] += 1

            # classifier
            for j in range(0, len(self.train_inputs)):
                dist_vect = distance(ex, self.train_inputs[j])
                if dist_vect < self.h:
                    counts[i][int(self.train_labels[j])-1] += 1
                    has_neighbours = True

            if has_neighbours:
                inv_sum = 1 / sum(kernel)
                classes_pred[i] = (inv_sum * counts[i]).argmax() + 1
            else:
                classes_pred[i] = draw_rand_label(ex, self.label_list)

        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = np.unique(train_labels)

    def compute_predictions(self, test_data):
        nbr_test = test_data.shape[0]
        classes_pred = np.zeros(nbr_test, dtype=int)
        const = 1 / ( ((2*math.pi)**(len(test_data[0])/2)) * (self.sigma**len(test_data[0])) )
        counts = np.ones((nbr_test, len(self.label_list)))

        # pour chaque point de test
        for (i, ex) in enumerate(test_data):
            rbf = np.zeros(len(self.train_inputs))

            # calculer l'inverse de la somme des kernels
            for j in range(0, len(self.train_inputs)):
                dist_2vect = distance(ex, self.train_inputs[j])
                exp = math.exp( (-1/2)*( (dist_2vect**2) / (self.sigma**2) ) )
                rbf[j] = const * exp

            # classifier
            for j in range(0, len(self.train_inputs)):
                counts[i][int(self.train_labels[j]) - 1] += rbf[j]

            inv_sum = 1 / sum(rbf)

            classes_pred[i] = (inv_sum * counts[i]).argmax() + 1

        return classes_pred


def distance(A, B):
    sum = 0
    for i in range(0, len(A)):
        sum += (A[i]-B[i])**2
    return np.sqrt(sum)

def split_dataset(iris):
    train = np.empty(shape=(0,5))
    validation = np.empty(shape=(0,5))
    test = np.empty(shape=(0,5))

    for i in range(len(iris)):
        if i % 5 == 0 or i % 5 == 1 or i % 5 == 2:
            train = np.append(train, [iris[i]], axis=0)
        elif i % 5 == 3:
            validation = np.append(validation, [iris[i]], axis=0)
        elif i % 5 == 4:
            test = np.append(test, [iris[i]], axis=0)

    sets = (train, validation, test)
    return sets


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        f = HardParzen(h)
        f.train(self.x_train, self.y_train)
        predictions = f.compute_predictions(self.x_val)
        error_count = 0

        # comparer les labels de 'predictions' et 'y_val'
        for i in range(0,len(predictions)):
            if predictions[i] != self.y_val[i]:
                error_count += 1

        return error_count/len(predictions)

    def soft_parzen(self, sigma):
        f = SoftRBFParzen(sigma)
        f.train(self.x_train, self.y_train)
        predictions = f.compute_predictions(self.x_val)
        error_count = 0

        # comparer les labels de 'predictions' et 'y_val'
        for i in range(0, len(predictions)):
            if predictions[i] != self.y_val[i]:
                error_count += 1

        return error_count/len(predictions)


def get_test_errors(iris):
    # pr??parer les donn??es
    sets = split_dataset(iris)
    x_train = sets[0][:, [0,1,2,3]]
    y_train = sets[0][:, [4]]
    x_val = sets[1][:, [0,1,2,3]]
    y_val = sets[1][:, [4]]
    test_error_val = ErrorRate(x_train, y_train, x_val, y_val)

    # valeurs de h et sigma ?? tester
    h_valeurs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    sigma_valeurs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

    global min_h, min_s
    min_h_err = math.inf
    min_s_err = math.inf

    for i in range(0, len(h_valeurs)):
        # taux d'erreurs sur l'ensemble validation
        errors_val_hp = test_error_val.hard_parzen(h_valeurs[i])
        errors_val_sp = test_error_val.soft_parzen(sigma_valeurs[i])
        if errors_val_hp < min_h_err:
            min_h_err = errors_val_hp
            min_h = h_valeurs[i]
        if errors_val_sp < min_s_err:
            min_s_err = errors_val_sp
            min_s = sigma_valeurs[i]

    # taux d???erreur sur l???ensemble de test
    x_test = sets[2][:, [0,1,2,3]]
    y_test = sets[2][:, [4]]
    test_error_test = ErrorRate(x_train, y_train, x_test, y_test)

    return [test_error_test.hard_parzen(min_h), test_error_test.soft_parzen(min_s)]


def random_projections(X, A):
    pass
