from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def linear_regression_model(config):
    return SGDRegressor(
        loss=config["loss"],
        max_iter=config["max_iter"]
    )


def l1_model():
    return LogisticRegression(penalty="l1", solver="liblinear")


def l2_model():
    return LogisticRegression(penalty="l2")


def softmax_model():
    return LogisticRegression(penalty="l2")


def svm_model(config):
    return SVC(
        C=config.get("C", 5),
        kernel=config.get("kernel", "linear")
    )

def knn_model(config):
    return KNeighborsClassifier(n_neighbors=config["n_neighbors"])
