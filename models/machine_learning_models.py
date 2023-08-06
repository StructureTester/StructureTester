import joblib
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_decision_tree_model(x_train, y_train, filename):
    clf = DecisionTreeClassifier()

    clf.fit(x_train, y_train)

    joblib.dump(clf, filename)


def train_logistic_regression_model(x_train, y_train, filename):
    clf = LogisticRegression()

    clf.fit(x_train, y_train)

    joblib.dump(clf, filename)


def train_random_forest_model(x_train, y_train, filename):
    clf = RandomForestClassifier()

    clf.fit(x_train, y_train)

    joblib.dump(clf, filename)


def train_svm_model(x_train, y_train, filename):
    clf = svm.SVC()

    clf.fit(x_train, y_train)

    joblib.dump(clf, filename)
