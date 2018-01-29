import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier


def MLP_CANCER():
    cancer = load_breast_cancer();

    print(cancer.DESCR)
    print(cancer.feature_names)
    print(cancer.target_names)
    print(cancer.data)
    #print(cancer.type)
    #print(cancer.data.shap)

    X_train, X_test, Y_train,Y_test = train_test_split(cancer.data,cancer.target,test_size=0.25)

    # Encode class labels as binary vector (with exactly ONE bit set to 1, and all others to 0)
    Y_train_OneHot = np.eye(2)[Y_train]
    Y_test_OneHot = np.eye(2)[Y_test]

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Plot the training points...
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright)
    plt.show()
    #   ...and testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], marker='x', c=Y_test, cmap=cm_bright, alpha=0.3)
    plt.show()

    print("Datasets: circles=training, light-crosses=test [and red=class_1, blue=class_2]")

    clf = MLPClassifier(hidden_layer_sizes=(1,), solver='sgd',
                        batch_size=4, learning_rate_init=0.005,
                        max_iter=500, shuffle=True)
    # Train the MLP classifier on training dataset
    clf.fit(X_train, Y_train_OneHot)

    print("Number of layers: ", clf.n_layers_)
    print("Number of outputs: ", clf.n_outputs_)

    print(clf.predict(X_train)[1:5, :])
    print(clf.predict_proba(X_train)[1:5, :])

    h = np.argmax(clf.predict(X_train), axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=cm_bright)
    ax[0].set_title("Data");
    ax[1].scatter(X_train[:, 0], X_train[:, 1], c=h, cmap=cm_bright)
    ax[1].set_title("Prediction");

def MLP_ECOLI():
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

    data = np.array(data)
    print(data)
    print(targets)