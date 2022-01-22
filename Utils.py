from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

class Utils:

    @staticmethod
    def grid_search_(X, Y, param_grid, scoring, kfold, model_, ttl):
        model = model_
        grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
        grid_result = grid.fit(X, Y)

        #Print Results
        print(ttl + ". Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        ranks = grid_result.cv_results_['rank_test_score']
        for mean, stdev, param, rank in zip(means, stds, params, ranks):
            print("#%d %f (%f) with: %r" % (rank, mean, stdev, param))

        return grid_result

    @staticmethod
    def plot_oos_results(df, ttl):
        fig, axes = plt.subplots(nrows=2, figsize=(10, 6), gridspec_kw={'height_ratios': [5, 1]})
        df[['cum_return', 'cum_strategy']].plot(ax=axes[0])
        df['prediction'].plot(ax=axes[1])

        fig.suptitle(ttl)
        plt.tight_layout()


    @staticmethod
    def show_confusion_matrix(Y, x, ttl):
        plt.figure()
        df_cm = pd.DataFrame(confusion_matrix(Y, x), columns=np.unique(Y), index = np.unique(Y))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font sizes
        plt.title(ttl)
