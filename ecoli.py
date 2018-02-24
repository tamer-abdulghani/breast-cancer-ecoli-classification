from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sb
from sklearn.ensemble import ExtraTreesClassifier
import pandas
from sklearn import tree
import pydotplus
import collections
from sklearn import svm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_decision_regions

import csv

def loadEcoliCSV(csvFile):

    data = []
    targets = []

    '''
    ecoli_labels = np.genfromtxt(csvFile, delimiter=',', dtype=str)
    print(set(ecoli_labels))
    ecoli_targets = np.zeros(ecoli_labels.size, dtype=int)
    labels = np.unique(ecoli_labels)
    for l in range(len(labels)):
        ecoli_targets[ecoli_labels == labels[l]] = l

    labels = set(ecoli_labels)
    '''
    features_names = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']

    with open(csvFile) as csvfile:
        datasetreader = csv.reader(csvfile, delimiter=',')
        for row in datasetreader:
            if len(row) == 9:
                data.append([float(i) for i in row[1:8]])
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

    dataset = {}
    dataset['data'] = data
    dataset['target'] = targets
    dataset['feature_names'] = features_names

    return dataset

def showPCA(dataset):
    pca = PCA(n_components=2)
    dataset_pca = pca.fit_transform(dataset['data'])
    plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1],
                c=dataset['target'], edgecolor='none', alpha=1,
                cmap=plt.cm.get_cmap('spectral', len(set(dataset['target']))))

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

def showHeatMap(dataset):
    pca = PCA(n_components=2)
    comps = pd.DataFrame(columns=dataset['feature_names'])
    print(comps)
    sb.heatmap(comps, annot=False, linewidths=.5)
    plt.show()


def rankMyFeatures(dataset):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(dataset['data'], dataset['target'])
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(dataset['data'].shape[1]):
        print(
        "%d. feature %d %s (%f)" % (f + 1, indices[f], dataset['feature_names'][indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(dataset['data'].shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(dataset['data'].shape[1]), indices)
    plt.xlim([-1, dataset['data'].shape[1]])
    plt.show()


def correlationOfFirst8Features(dataset,csvFile):
    arr = np.append(dataset['feature_names'], 'index')
    data2 = pandas.read_csv(csvFile, names=arr)

    g = sb.pairplot(data2, kind="scatter", hue='index')
    plt.show()


def mlpClassifierTrainTest(dataset):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.10)

    # One Hot encoding
    Y_train_OneHot = np.eye(9)[Y_train]
    Y_test_OneHot = np.eye(9)[Y_test]

    clf = MLPClassifier(activation='relu', alpha=1, batch_size='auto', hidden_layer_sizes=(10, 15),
                        learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, random_state=1, shuffle=True, solver='lbfgs')
    clf.fit(X_train, Y_train_OneHot)
    print("Number of layers: ", clf.n_layers_)
    print("Number of outputs: ", clf.n_outputs_)
    h_train_pred = np.argmax(clf.predict(X_train), axis=1)
    h_test_pred = np.argmax(clf.predict(X_test), axis=1)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
    plt.title("Real labels [train]")
    plt.subplot(2, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=h_train_pred)
    plt.title("MLP [train]")
    plt.subplot(2, 2, 3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    plt.title("Real labels [test]")
    plt.subplot(2, 2, 4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=h_test_pred)
    plt.title("MLP [test]");

    print("Train accuracy: ", clf.score(X_train, Y_train_OneHot), " test accuracy:", clf.score(X_test, Y_test_OneHot))

    plt.show()

def dtClassifierTrainTest(dataset):
    X_train, X_test, Y_train, Y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.25)

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X_train, Y_train)

    train_labels = classifier.predict(X_train)
    test_labels = classifier.predict(X_test)

    print('Score: {}'.format(classifier.score(X_train, Y_train)))
    print('Score: {}'.format(classifier.score(X_test, Y_test)))

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train)
    plt.title("Real labels [train]")
    plt.subplot(2, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels)
    plt.title("DT [train]")
    plt.subplot(2, 2, 3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
    plt.title("Real labels [test]")
    plt.subplot(2, 2, 4)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels)
    plt.title("DT [test]")
    plt.show()

def visualizeDecisionTree(dataset):

    X_train, X_test, Y_train, Y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.25)
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X_train, Y_train)

    dot_data = tree.export_graphviz(classifier,
                                    feature_names=dataset['feature_names'],
                                    out_file=None,
                                    filled=True
                                    )

    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('orange', 'green')
    edges = collections.defaultdict(list)

    # print(dot_data)

    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])

    graph.write_png('ecoli_dt.png')

def svmClassifierTrainTest(dataset):
    X_train_cancer, X_test_cancer, Y_train_cancer, Y_test_cancer = train_test_split(dataset['data'], dataset['target'],
                                                                                    test_size=0.20)

    clf = svm.SVC()
    clf.fit(X_train_cancer, Y_train_cancer)

    # Plot training+testing dataset
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the training points
    plt.scatter(X_train_cancer[:, 0], X_train_cancer[:, 1], c=Y_train_cancer, cmap=cm_bright)
    # Plot the testing points
    plt.scatter(X_test_cancer[:, 0], X_test_cancer[:, 1], marker='x', c=Y_test_cancer, cmap=cm_bright, alpha=0.9)

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_cancer[:, 0], X_train_cancer[:, 1], c=Y_train_cancer, s=50, cmap='viridis')
    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_cancer[:, 0], X_train_cancer[:, 1], c=Y_train_cancer, s=50, cmap='viridis')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.9)

    predicted_label = clf.predict(X_test_cancer)
    plt.show()

def svmVisualizeTwoFeatures(dataset):
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(dataset['data']), np.array(dataset['target']), test_size=0.20)

    model = svm.SVC(kernel='rbf', C=1)

    X = X_train[:, [0, 1]]
    model.fit(X, Y_train)
    plot_decision_regions(X=X,
                          y=Y_train,
                          clf=model,
                          legend=2)
    plt.show()


if __name__ == '__main__':
    ecoli_csv = 'dataset/Ecoli/ecoli-dataset.csv';
    dataset = loadEcoliCSV(ecoli_csv)
    showPCA(dataset)
    #showHeatMap(dataset)
    #rankMyFeatures(dataset)
    #correlationOfFirst8Features(dataset,ecoli_csv)
    #mlpClassifierTrainTest(dataset)
    #dtClassifierTrainTest(dataset)
    #visualizeDecisionTree(dataset)
    #svmClassifierTrainTest(dataset)
    #svmVisualizeTwoFeatures(dataset)