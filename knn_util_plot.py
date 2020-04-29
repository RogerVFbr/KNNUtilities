import pandas as pd

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


class KNNUtilPlot:

    def __init__(self):
        sns.set()

    @staticmethod
    def print_influence_study(k_results):

        df = pd.DataFrame(k_results)
        df.sort_values(by=['k', 'col', 'weight'], inplace=True)
        cols_in_data = 2*(df.col.max()+1)
        bar_width = 0.9

        fig, ax = plt.subplots(2, figsize=(15, 15))

        sr_a = df[['k', 'col', 'weight', 'avg_accuracy']].groupby(['k', 'col', 'weight']).sum()
        index = [f"K={df.iloc[x].k} {df.iloc[x].col} {df.iloc[x].weight[:1].upper()}" if x%cols_in_data == 0 else
                 f"{df.iloc[x].col} {df.iloc[x].weight[:1].upper()}" for x in range(len(sr_a))]

        sr_a.plot.bar(
            figsize=(35,25),
            legend=False,
            width=bar_width,
            color='bbggrrccmmyykkwwbbggrrccmmyykkww'[:cols_in_data],
            ylim=(
                df.avg_accuracy.min()*0.95,
                df.avg_accuracy.max()*1.05
            ),
            ax=ax[0]
        )
        ax[0].set_xlabel('')
        ax[0].set_ylabel('')
        ax[0].set_title('KNN Accuracy per column', fontsize=30, pad=20)
        ax[0].set_xticks(range(len(index)))
        ax[0].set_xticklabels(index, fontsize=15)
        ax[0].tick_params(axis="y", labelsize=20)

        sr_b = df[['k', 'col', 'weight', 'variance']].groupby(['k', 'col', 'weight']).sum()
        sr_b.plot.bar(
            legend=False,
            width=bar_width,
            color='bbggrrccmmyykkwwbbggrrccmmyykkww'[:cols_in_data],
            ylim=(
                df.variance.min()*0.8,
                df.variance.max()*1.05,
            ),
            ax=ax[1]
        )
        ax[1].set_xlabel('')
        ax[1].set_ylabel('')
        ax[1].set_title('KNN Variance per column', fontsize=30, pad=20)
        ax[1].set_xticks(range(len(index)))
        ax[1].set_xticklabels(index, fontsize=15)
        ax[1].tick_params(axis="y", labelsize=20)

        for x in range(cols_in_data):
            ax[0].axvline(x=x*cols_in_data-bar_width/2, c='k', linewidth=2)
            ax[1].axvline(x=x*cols_in_data-bar_width/2, c='k', linewidth=2)

        plt.show()

    @staticmethod
    def print_k_study(title, k_results):

        groups = [x['k'] for x in k_results if x['weight'] == 'uniform']

        accuracy_uniform = [x['avg_accuracy'] for x in k_results if x['weight'] == 'uniform']
        accuracy_distance = [x['avg_accuracy'] for x in k_results if x['weight'] == 'distance']
        variance_uniform = [x['variance'] for x in k_results if x['weight'] == 'uniform']
        variance_distance = [x['variance'] for x in k_results if x['weight'] == 'distance']

        fig, ax = plt.subplots(2, figsize=(15, 15))

        index = np.arange(len(groups))
        bar_width = 0.4
        opacity = 1

        ax[0].bar(index, accuracy_uniform, bar_width,
                  alpha=opacity,
                  color='b',
                  label='uniform')

        ax[0].bar(index + bar_width, accuracy_distance, bar_width,
                  alpha=opacity,
                  color='g',
                  label='distance')

        ylim_bot = min(min(accuracy_uniform), min(accuracy_distance))*0.95
        ylim_top = max(max(accuracy_uniform), max(accuracy_distance))*1.05

        ax[0].set_ylim(ylim_bot if ylim_bot >= 0 else 0, ylim_top if ylim_top <= 100 else 100)
        ax[0].set_xlabel('K Value')
        ax[0].set_ylabel('')
        ax[0].set_title(title, fontdict={'fontsize': 25}, pad=15)
        ax[0].set_xticks(index + bar_width / 2)
        ax[0].set_xticklabels(groups)
        ax[0].legend()

        ax[1].bar(index, variance_uniform, bar_width,
                  alpha=opacity,
                  color='r',
                  label='uniform')

        ax[1].bar(index + bar_width, variance_distance, bar_width,
                  alpha=opacity,
                  color='y',
                  label='distance')

        ylim_bot = min(min(variance_uniform), min(variance_distance))
        ylim_top = max(max(variance_uniform), max(variance_distance))

        ylim_bot *= 0.9
        ylim_top *= 1.1

        ax[1].set_ylim(ylim_bot if ylim_bot >= 0 else 0, ylim_top if ylim_top <= 100 else 100)
        ax[1].set_xlabel('K Value')
        ax[1].set_ylabel('Variance')
        ax[1].set_title('Variance / K', fontdict={'fontsize': 25}, pad=15)
        ax[1].set_xticks(index + bar_width / 2)
        ax[1].set_xticklabels(groups)
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        k_results = sorted(k_results, key=lambda i: i['avg_accuracy'], reverse=True)

    @staticmethod
    def knn_base_example(self):
        n_neighbors = 15

        # import some data to play with
        iris = datasets.load_iris()

        # we only take the first two features. We could avoid this ugly
        # slicing by using a two-dim dataset
        X = iris.data[:, :2]
        y = iris.target

        h = .02  # step size in the mesh

        # Create color maps
        cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

        for weights in ['uniform', 'distance']:
            # we create an instance of Neighbours Classifier and fit the data.
            clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
            clf.fit(X, y)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure()
            plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title("3-Class classification (k = %i, weights = '%s')"
                      % (n_neighbors, weights))

        plt.show()

