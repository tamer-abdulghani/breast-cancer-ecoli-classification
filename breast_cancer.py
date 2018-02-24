from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import pandas
from sklearn import tree
import pydotplus
import collections
from sklearn import svm
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions

def showPCA():
    dataset = load_breast_cancer()
    pca = PCA(n_components=2)
    dataset_pca = pca.fit_transform(dataset.data)
    plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1],
                c=dataset.target, edgecolor='none', alpha=1,
                cmap=plt.cm.get_cmap('spectral', 2))

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()

def showHeatMap():
    dataset = load_breast_cancer()
    pca = PCA(n_components=2)
    comps = pd.DataFrame(pca.components_, columns=dataset.feature_names)
    print(comps)
    sb.heatmap(comps, annot=False, linewidths=.5)
    plt.show()


def rankMyFeatures():
    dataset = load_breast_cancer()
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(dataset.data, dataset.target)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(dataset.data.shape[1]):
        print(
        "%d. feature %d %s (%f)" % (f + 1, indices[f], dataset.feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(dataset.data.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(dataset.data.shape[1]), indices)
    plt.xlim([-1, dataset.data.shape[1]])
    plt.show()


def correlationOfFirst8Features():
    csvFile = 'dataset/BreastCancer/cancer.csv'
    dataset = load_breast_cancer()
    arr = np.append(dataset.feature_names, 'index')
    data2 = pandas.read_csv(csvFile, names=arr)
    newData = data2[
        ['worst concave points',
         'worst radius',
         'worst area',
         'worst perimeter',
         'mean concave points',
         'mean radius',
         'index'
         ]
    ]
    g = sb.pairplot(newData, kind="scatter", hue='index')
    plt.show()


def mlpClassifier():
    dataset = load_breast_cancer()
    X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.10)

    # One Hot encoding
    Y_train_OneHot = np.eye(2)[Y_train]
    Y_test_OneHot = np.eye(2)[Y_test]

    clf = MLPClassifier(hidden_layer_sizes=(3,), solver='lbfgs',
                        batch_size=4, alpha=1,
                        max_iter=5000, shuffle=True)
    clf.fit(X_train, Y_train_OneHot)
    print("Number of layers: ", clf.n_layers_)
    print("Number of outputs: ", clf.n_outputs_)
    h = np.argmax(clf.predict(X_train), axis=1)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright)
    ax[0].set_title("Data");
    ax[1].scatter(X_train[:, 0], X_train[:, 1], c=h, cmap=cm_bright)
    ax[1].set_title("Prediction");
    print("Train accuracy: ", clf.score(X_train, Y_train_OneHot), " test accuracy:", clf.score(X_test, Y_test_OneHot))
    plt.show()


def mlpClassifierTrainTest():
    cancer = load_breast_cancer();

    X = cancer.data
    y = cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    # Encode class labels as binary vector (with exactly ONE bit set to 1, and all others to 0)
    Y_train_OneHot = np.eye(2)[y_train]
    Y_test_OneHot = np.eye(2)[y_test]

    # act: logistic   , alpha:  0.0316227766017 , layers:  [5, 5] , solver:  lbfgs
    clf = MLPClassifier(hidden_layer_sizes=(10, 3), solver='lbfgs',
                        batch_size=4, alpha=1,
                        max_iter=200, shuffle=True)
    clf.fit(X_train, Y_train_OneHot)

    h_train_pred = np.argmax(clf.predict(X_train), axis=1)
    h_test_pred = np.argmax(clf.predict(X_test), axis=1)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlabel(cancer.feature_names[0])
    plt.ylabel(cancer.feature_names[1])
    plt.title("Real labels [train]")

    plt.subplot(2, 2, 2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=h_train_pred)
    plt.xlabel(cancer.feature_names[0])
    plt.ylabel(cancer.feature_names[1])
    plt.title("MLP [train]")

    plt.subplot(2, 2, 3)
    plt.xlabel(cancer.feature_names[0])
    plt.ylabel(cancer.feature_names[1])
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    plt.title("Real labels [test]")

    plt.subplot(2, 2, 4)
    plt.xlabel(cancer.feature_names[0])
    plt.ylabel(cancer.feature_names[1])
    plt.scatter(X_test[:, 0], X_test[:, 1], c=h_test_pred)
    plt.title("MLP [test]");

    print("Train accuracy: ", clf.score(X_train, Y_train_OneHot), " test accuracy:", clf.score(X_test, Y_test_OneHot))
    plt.show()

def dtClassifierTrainTest():
    cancer = load_breast_cancer();
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.25)

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

def visualizeDecisionTree():
    cancer = load_breast_cancer();
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.25)
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(X_train, Y_train)

    dot_data = tree.export_graphviz(classifier,
                                    feature_names=cancer.feature_names,
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

    graph.write_png('breast_cancer_dt.png')

def svmClassifierTrainTest():
    cancer = load_breast_cancer();
    X_train_cancer, X_test_cancer, Y_train_cancer, Y_test_cancer = train_test_split(cancer.data, cancer.target,
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

def svmVisualizeTwoFeature():
    cancer = load_breast_cancer();
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.20)

    model = SVC(kernel='rbf', C=1)

    X = X_train[:, [0, 1]]
    X2 = X_test[:, [0, 1]]
    model.fit(X, Y_train)
    print("Train acc: ", model.score(X, Y_train), ", Test Acc:", model.score(X2, Y_test))
    plot_decision_regions(X=X,
                          y=Y_train,
                          clf=model,
                          legend=2)
    plt.show()


if __name__ == '__main__':
    showPCA()
    #showHeatMap()
    #rankMyFeatures()
    #correlationOfFirst8Features()
    #mlpClassifier()
    #mlpClassifierTrainTest()
    #dtClassifierTrainTest()
    #visualizeDecisionTree()
    #svmClassifierTrainTest()
    #svmVisualizeTwoFeature()