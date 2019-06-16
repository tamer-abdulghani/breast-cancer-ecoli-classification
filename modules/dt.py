import csv

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def cancer_dt():
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25)

    classifier = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                             max_features=None, max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, min_impurity_split=None,
                                             min_samples_leaf=1, min_samples_split=2,
                                             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                             splitter='best')
    classifier.fit(x_train, y_train)

    predicao = classifier.predict(x_test)
    print(classifier)
    print('Score: {}'.format(classifier.score(x_train, y_train)))
    print('Score: {}'.format(classifier.score(x_test, y_test)))

    # graph
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
    plt.show()

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright)
    plt.show()


def ecoli_dt():
    data = []
    targets = []
    ecoli_csv = '../datasets/ecoli/ecoli-datasets.csv'

    with open(ecoli_csv, newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if len(row) == 9:
                print(row[1:8])
                data.append([i for i in row[1:8]])
                if row[8] == "cp":
                    targets.append(1)
                elif row[8] == "im":
                    targets.append(2)
                elif row[8] == "pp":
                    targets.append(3)
                elif row[8] == "imU":
                    targets.append(4)
                elif row[8] == "om":
                    targets.append(5)
                elif row[8] == "omL":
                    targets.append(6)
                elif row[8] == "imL":
                    targets.append(7)
                else:
                    targets.append(8)

    data = np.array(data)

    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)

    classifier = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                             max_features=None, max_leaf_nodes=None,
                                             min_impurity_decrease=0.0, min_impurity_split=None,
                                             min_samples_leaf=1, min_samples_split=2,
                                             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                                             splitter='best')
    classifier.fit(x_train, y_train)
    print(classifier)
    predicao = classifier.predict(x_test)

    print('Score: {}'.format(classifier.score(x_train, y_train)))
    print('Score: {}'.format(classifier.score(x_test, y_test)))

    # graph
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
    plt.show()

    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright)
    plt.show()


cancer_dt()
# ecoli_dt()
