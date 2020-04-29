import math
import seaborn as sns
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from knn_util_data_cleaning import KNNUtilDataCleaning
from knn_util_plot import KNNUtilPlot


class KNNUtil:

    def __init__(self):
        sns.set()

    @staticmethod
    def get_column_influence(X, y, k_range, no_of_folds, regressor=False):
        k_results = []
        X_full = X.copy()

        if type(X) is not np.ndarray:
            X_full = X_full.to_numpy()

        for i in range(0, X.shape[1]):
            X = X_full[:, i:i+1]
            for k in range(k_range[0], k_range[1] + 1):
                for weight in ['uniform', 'distance']:
                    results = []
                    for random_state in range(no_of_folds):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                                            random_state=random_state)
                        if not regressor:
                            clf = neighbors.KNeighborsClassifier(k, weights=weight)
                        else:
                            clf = neighbors.KNeighborsRegressor(k, weights=weight)
                        clf.fit(X_train, y_train)
                        Z = clf.predict(X_test)
                        # accuracy = round(len([x for x, y in zip(Z, y_test) if x == y]) / len(Z) * 100, 3)
                        if not regressor:
                            accuracy = round(len([x for x, y in zip(Z, y_test) if x == y]) / len(Z) * 100, 3)
                        else:
                            accuracy = [x - y for x, y in zip(Z, y_test)]
                        results.append(accuracy)
                        if random_state % 100 == 0:
                            print(f" COL: {i} | K: {k} | W: {weight[:3]} | RND: {random_state}")

                    k_results.append({
                        'col': i,
                        'k': k,
                        'weight': weight,
                        'avg_accuracy': np.mean(results),
                        'variance': np.var(results)
                    })

        KNNUtilPlot().print_influence_study(k_results)

    @classmethod
    def get_k(cls, X, y, k_range, no_of_folds, regressor=False):
        k_results = []
        for k in range(k_range[0], k_range[1]+1):
            for weight in ['uniform', 'distance']:
                results =[]
                for random_state in range(no_of_folds):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
                    if not regressor:
                        clf = neighbors.KNeighborsClassifier(k, weights=weight)
                    else:
                        clf = neighbors.KNeighborsRegressor(k, weights=weight)
                    clf.fit(X_train, y_train)
                    Z = clf.predict(X_test)
                    if not regressor:
                        accuracy = round(len([x for x, y in zip(Z, y_test) if x == y])/len(Z)*100, 3)
                    else:
                        accuracy = cls.rmse(Z, y_test)
                    results.append(accuracy)
                    if random_state%100 == 0:
                        print(f" K: {k} | W: {weight[:3]} | RND: {random_state}")

                k_results.append({
                    'k': k,
                    'weight': weight,
                    'avg_accuracy': np.mean(results),
                    'variance': np.var(results)
                })

        if not regressor:
            title = 'Accuracy % / K'
        else:
            title = 'Deviation / K'

        KNNUtilPlot().print_k_study(title, k_results)

    @staticmethod
    def rmse(x, y):
        return math.sqrt(((x - y) ** 2).mean())


if __name__ == "__main__":

    # iris = datasets.load_iris()
    # data = iris.data[:, :]
    # target = iris.target

    kaggle_house = KNNUtilDataCleaning().prepare('data_sets/kaggle_house/train.csv')
    data = kaggle_house.copy().drop(['SalePrice', 'Id'], axis=1).to_numpy()[:, :]
    target = np.log(kaggle_house.copy().SalePrice).to_numpy()

    ku = KNNUtil()
    ku.get_k(X=data, y=target, k_range=(5, 35), no_of_folds=1, regressor=True)
    # ku.get_column_influence(X=data, y=target, k_range=(10, 20), no_of_folds=50, regressor=True)
