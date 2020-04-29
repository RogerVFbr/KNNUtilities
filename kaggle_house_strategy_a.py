import math
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from param_shuffler import ParamShuffler
from knn_util_data_cleaning import KNNUtilDataCleaning


class KaggleHouseStrategyA:

    def __init__(self, path):
        self.train = pd.read_csv(path)
        self.X = KNNUtilDataCleaning().prepare(self.train).drop(['SalePrice', 'Id'], axis=1)
        self.y = np.log(self.train.copy().SalePrice)

    def fit(self, seed, n, k, p, algo, weight):
        corr_train = self.train.corr()
        top_n_cols = corr_train.nlargest(n, 'SalePrice')['SalePrice'].index
        X_train, X_test, y_train, y_test = train_test_split(self.X[top_n_cols[1:]], self.y, test_size=0.30,
                                                            random_state=seed)
        clf = KNeighborsRegressor(k, weights=weight, p=p, algorithm=algo)
        clf.fit(X_train, y_train)
        res = [self.rmse(clf.predict(X_train), y_train), self.rmse(clf.predict(X_test), y_test)]
        return res[1]

    @staticmethod
    def print_results(results):
        df = pd.DataFrame(results)
        no_of_unique = df.seed.nunique()
        final_result = df \
            .groupby(['k', 'n', 'p', 'algo', 'weight']) \
            .agg({'result': ['mean', 'var', 'min', 'max']})

        print('.')
        print(f'HIGHEST MEANS ({no_of_unique} seeds, {df.shape[0]} iterations):')
        print(final_result.sort_values(by=[('result', 'mean')]).head(10))
        print()
        print(f'SMALLEST VARIANCES ({no_of_unique} seeds, {df.shape[0]} iterations):')
        print(final_result.sort_values(by=[('result', 'var')]).head(10))
        print()

    @staticmethod
    def rmse(x, y):
        return math.sqrt(((x - y) ** 2).mean())


if __name__ == "__main__":

    strat = KaggleHouseStrategyA("data_sets/kaggle_house/train.csv")

    ps = ParamShuffler(strat.fit)
    results = ps.run(
        OrderedDict({
            'seed': range(40, 80),
            'n': range(8,19),
            'k': range(7, 25),
            'p': [1, 2],
            'algo': ['ball_tree', 'kd_tree', 'brute'],
            'weight': ['distance', 'uniform']
        })
    )

    strat.print_results(results)