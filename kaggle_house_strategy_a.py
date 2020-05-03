import math
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

    def logic(self, seed, n, k, p, algo, weight):

        corr_train = self.train.corr()
        top_n_cols = corr_train.nlargest(n, 'SalePrice')['SalePrice'].index

        X_train, X_test, y_train, y_test = train_test_split(self.X[top_n_cols[1:]], self.y, test_size=0.30,
                                                            random_state=seed)

        clf = KNeighborsRegressor(k, weights=weight, p=p, algorithm=algo)
        clf.fit(X_train, y_train)
        return self.rmse(clf.predict(X_test), y_test)

    @staticmethod
    def print_results(results):
        df = pd.DataFrame(results)
        final_result = df \
            .groupby(['k', 'n', 'p', 'algo', 'weight']) \
            .agg({'result': ['mean', 'var', 'min', 'max']})

        print('.')
        print('PARAMETERS:')
        for x in df.columns:
            if x == 'result': continue
            print(x, df[x].unique())
        print()
        print(f'HIGHEST MEANS ({df.seed.nunique()} seeds, {df.shape[0]} iterations):')
        print(final_result.sort_values(by=[('result', 'mean')]).head(10))
        print()
        print(f'SMALLEST VARIANCES ({df.seed.nunique()} seeds, {df.shape[0]} iterations):')
        print(final_result.sort_values(by=[('result', 'var')]).head(10))
        print()

    @staticmethod
    def rmse(x, y):
        return math.sqrt(((x - y) ** 2).mean())


if __name__ == "__main__":

    strat = KaggleHouseStrategyA("data_sets/kaggle_house/train.csv")

    ps = ParamShuffler(strat.logic)
    results = ps.run({
        'seed': range(40, 42),
        'n': range(4, 6),
        'k': range(9, 11),
        'p': [1],
        'algo': ['ball_tree', 'kd_tree', 'brute'],
        'weight': ['distance', 'uniform']
    })

    ps.save_to_csv(results, file_name='kaggle_house')

    strat.print_results(results)
