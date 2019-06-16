import collections
import csv

import numpy as np
import pandas
import pandas as pd
import pydotplus
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def load_ecoli_dataset(file_path):
    data = []
    targets = []
    features_names = ['mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2']
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
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


def show_pca(dataset):
    pca = PCA(n_components=2)
    dataset_pca = pca.fit_transform(dataset['data'])
    plt.scatter(dataset_pca[:, 0], dataset_pca[:, 1],
                c=dataset['target'], edgecolor='none', alpha=1,
                cmap=plt.cm.get_cmap('spectral', len(set(dataset['target']))))

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()


def show_heat_map(dataset):
    pca = PCA(n_components=2)
    comps = pd.DataFrame(columns=dataset['feature_names'])
    print(comps)
    sb.heatmap(comps, annot=False, linewidths=.5)
    plt.show()


def rank_my_feature(dataset):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(dataset['data'], dataset['target'])
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(dataset['data'].shape[1]):
        print(
            "%d. feature %d %s (%f)" % (
                f + 1, indices[f], dataset['feature_names'][indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(dataset['data'].shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(dataset['data'].shape[1]), indices)
    plt.xlim([-1, dataset['data'].shape[1]])
    plt.show()


def correlation_of_first_8_features(dataset, csvFile):
    arr = np.append(dataset['feature_names'], 'index')
    data2 = pandas.read_csv(csvFile, names=arr)

    g = sb.pairplot(data2, kind="scatter", hue='index')
    plt.show()


def mlp_cassifier(dataset):
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


def dt_classifier(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.25)

    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)

    train_labels = classifier.predict(x_train)
    test_labels = classifier.predict(x_test)

    print('Score: {}'.format(classifier.score(x_train, y_train)))
    print('Score: {}'.format(classifier.score(x_test, y_test)))

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.title("Real labels [train]")
    plt.subplot(2, 2, 2)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=train_labels)
    plt.title("DT [train]")
    plt.subplot(2, 2, 3)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test)
    plt.title("Real labels [test]")
    plt.subplot(2, 2, 4)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=test_labels)
    plt.title("DT [test]")
    plt.show()


def visualizeDecisionTree(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'], test_size=0.25)
    classifier = tree.DecisionTreeClassifier()
    classifier = classifier.fit(x_train, y_train)

    dot_data = tree.export_graphviz(classifier,
                                    feature_names=dataset['feature_names'],
                                    out_file=None,
                                    filled=True
                                    )

    graph = pydotplus.graph_from_dot_data(dot_data)

    colors = ('orange', 'green')
    edges = collections.defaultdict(list)
    for edge in graph.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(2):
            dest = graph.get_node(str(edges[edge][i]))[0]
            dest.set_fillcolor(colors[i])
    graph.write_png('ecoli_dt.png')


def svmClassifierTrainTest(dataset):
    x_train_cancer, x_test_cancer, y_train_cancer, y_test_cancer = train_test_split(dataset['data'], dataset['target'],
                                                                                    test_size=0.20)
    clf = svm.SVC()
    clf.fit(x_train_cancer, y_train_cancer)
    # Plot training+testing datasets
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the training points
    plt.scatter(x_train_cancer[:, 0], x_train_cancer[:, 1], c=y_train_cancer, cmap=cm_bright)
    # Plot the testing points
    plt.scatter(x_test_cancer[:, 0], x_test_cancer[:, 1], marker='x', c=y_test_cancer, cmap=cm_bright, alpha=0.9)

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_train_cancer[:, 0], x_train_cancer[:, 1], c=y_train_cancer, s=50, cmap='viridis')
    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_train_cancer[:, 0], x_train_cancer[:, 1], c=y_train_cancer, s=50, cmap='viridis')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.9)

    predicted_label = clf.predict(x_test_cancer)
    plt.show()


def svmVisualizeTwoFeatures(dataset):
    x_train, x_test, y_train, y_test = train_test_split(np.array(dataset['data']), np.array(dataset['target']),
                                                        test_size=0.20)

    model = svm.SVC(kernel='rbf', C=1)

    X = x_train[:, [0, 1]]
    model.fit(X, y_train)
    plot_decision_regions(X=X,
                          y=y_train,
                          clf=model,
                          legend=2)
    plt.show()


if __name__ == '__main__':
    ecoli_csv = 'datasets/ecoli/ecoli-datasets.csv';
    dataset = load_ecoli_dataset(ecoli_csv)
    show_pca(dataset)
    # show_heat_map(datasets)
    # rankMyFeatures(datasets)
    # correlation_of_first_8_features(datasets,ecoli_csv)
    # mlp_cassifier(datasets)
    # dt_classifier(datasets)
    # visualizeDecisionTree(datasets)
    # svmClassifierTrainTest(datasets)
    # svmVisualizeTwoFeatures(datasets)
