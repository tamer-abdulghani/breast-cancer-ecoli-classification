
import csv
#%matplotlib inline
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.svm import SVC # "Support vector classifier"
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import csv
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.svm import SVC  # "Support vector classifier"
from sklearn.svm import SVC
from scipy import stats
from mlxtend.plotting import plot_decision_regions
from ipywidgets import interact, fixed

def SVM_breast_cancer():
    cancer = load_breast_cancer();
    X_train_cancer, X_test_cancer, Y_train_cancer,Y_test_cancer = train_test_split(cancer.data,cancer.target,test_size=0.20)

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
    plt.figure(figsize=(10,5))
    plt.scatter(X_train_cancer[:, 0], X_train_cancer[:, 1], c=Y_train_cancer, s=50, cmap='viridis')
    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10,5))
    plt.scatter(X_train_cancer[:, 0], X_train_cancer[:, 1], c=Y_train_cancer, s=50, cmap='viridis')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.9)

    predicted_label = clf.predict(X_test_cancer)
    print(accuracy_score(Y_test_cancer, predicted_label))
    print(precision_score(Y_test_cancer, predicted_label,  average="macro"))
    print(recall_score(Y_test_cancer, predicted_label,  average="macro"))
    print(f1_score(Y_test_cancer, predicted_label,  average="macro"))

def SVM_ecoli():
    data = []
    targets = []
    ecoli_csv = 'dataset/Ecoli/ecoli-dataset.csv';

    with open(ecoli_csv, newline='') as csvfile:
        datasetreader = csv.reader(csvfile, delimiter=',')
        for row in datasetreader:
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

    ecoli = np.array(data)
    X_train_ecoli, X_test_ecoli, Y_train_ecoli, Y_test_ecoli = train_test_split(ecoli.data, ecoli.target, test_size=0.20)
    print(data)
    print(targets)
    clf = svm.SVC()
    clf.fit(X_train_ecoli, Y_train_ecoli)

    # Plot training+testing dataset
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the training points
    plt.scatter(X_train_ecoli[:, 0], X_train_ecoli[:, 1], c=Y_train_ecoli, cmap=cm_bright)
    # Plot the testing points
    plt.scatter(X_test_ecoli[:, 0], X_test_ecoli[:, 1], marker='x', c=Y_test_ecoli, cmap=cm_bright, alpha=0.9)

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_ecoli[:, 0], X_train_ecoli[:, 1], c=Y_train_ecoli, s=50, cmap='viridis')
    for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
        plt.plot(xfit, m * xfit + b, '-k')

    # Fitting a SVM
    xfit = np.linspace(-1, 3.5)
    plt.figure(figsize=(10, 5))
    plt.scatter(X_train_ecoli[:, 0], X_train_ecoli[:, 1], c=Y_train_ecoli, s=50, cmap='viridis')

    for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-k')
        plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.9)

    predicted_label = clf.predict(X_test_ecoli)
    print(accuracy_score(Y_test_ecoli, predicted_label))
    print(precision_score(Y_test_ecoli, predicted_label, average="macro"))
    print(recall_score(Y_test_ecoli, predicted_label, average="macro"))
    print(f1_score(Y_test_ecoli, predicted_label, average="macro"))

def SVM_plotting():
    cancer = load_breast_cancer();
    X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.20)

    model = SVC(kernel='sigmoid', C=1)


    X = X_train[:,[0,1]]
    X2 = X_test[:, [0, 1]]
    model.fit(X, Y_train)
    print("Train acc: ", model.score(X, Y_train), ", Test Acc:", model.score(X2, Y_test))
    plot_decision_regions(X=X,
                          y=Y_train,
                          clf=model,
                          legend=2)
    plt.show()


def SVM_plotting_ecoli():
    data = []
    targets = []
    ecoli_csv = 'ecoli-dataset.csv';

    with open(ecoli_csv) as csvfile:
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

    ecoli = np.array(data)
    ecoli_labels = np.array(targets)

    X_train, X_test, Y_train, Y_test = train_test_split(ecoli, ecoli_labels, test_size=0.20)
    #X = X_train[:, [0, 1]]
    #X2 = X_test[:, [0, 1]]
    model = SVC(kernel='rbf', C=25)
    model.fit(X_train, Y_train)
    values = [-4.0, -1.0, 1.0, 4.0]
    width = 0.75
    fig, axarr = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for value, ax in zip(values, axarr.flat):
        plot_decision_regions(X=X_train, y=Y_train, clf=model, legend=2,
                              filler_feature_values={2: value},
                              filler_feature_ranges={2: width})

    plt.show()
    print("Train acc: ", model.score(X_train, Y_train), ", Test Acc:", model.score(X_test, Y_test))

    '''
    for i in range(1,10):
        for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
            X_train, X_test, Y_train, Y_test = train_test_split(ecoli, ecoli_labels, test_size=0.20)
            X = X_train[:, [0 ,1, 2, 3, 4, 5, 6]]
            X2 = X_test[:, [0, 1, 2, 3, 4, 5, 6]]
            model = SVC(kernel=kernel, C=i)
            model.fit(X, Y_train)
            plot_decision_regions(X=X,y=Y_train,clf=model,legend=2)
            plt.show()
            print("Train acc: ",model.score(X,Y_train),", Test Acc:",model.score(X2,Y_test))
    '''

def plot3D():
    from mpl_toolkits import mplot3d
    data = []
    targets = []
    ecoli_csv = 'ecoli-dataset.csv';

    with open(ecoli_csv) as csvfile:
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

    ecoli = np.array(data)
    ecoli_labels = np.array(targets)

    X_train, X_test, Y_train, Y_test = train_test_split(ecoli, ecoli_labels, test_size=0.20)
    r = np.exp(-(X_train ** 2).sum(1))

    def plot_3D(elev=30, azim=30, X=X_train, y=Y_train):
        ax = plt.subplot(projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap=plt.cm.get_cmap('spectral', 10))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('r')

    interact(plot_3D, elev=[-90, 90], azip=(-180, 180),X=fixed(X_train), y=fixed(Y_train));
    plt.show()

def bestAcc():
    data = []
    targets = []
    ecoli_csv = 'ecoli-dataset.csv';

    with open(ecoli_csv) as csvfile:
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

    ecoli = np.array(data)
    ecoli_labels = np.array(targets)

    X_train, X_test, Y_train, Y_test = train_test_split(ecoli, ecoli_labels, test_size=0.20)
    model = SVC(kernel='rbf', C=25)
    model.fit(X_train, Y_train)

    plt.show()
    print("Train acc: ", model.score(X_train, Y_train), ", Test Acc:", model.score(X_test, Y_test))


if __name__ == '__main__':
    #SVM_plotting_ecoli()
    #SVM_plotting()
    #SVM_ecoli()
    #plot3D()
    bestAcc()