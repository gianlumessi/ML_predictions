from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from Data_manager import Data_manager
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn_genetic import GASearchCV, GAFeatureSelectionCV
from sklearn_genetic.callbacks import ConsecutiveStopping
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn_genetic.space import Integer, Continuous, Categorical
from sklearn import tree


def plot_tree_(model, cols):
    _ = plt.figure(figsize=(10, 6))
    _ = tree.plot_tree(model, feature_names=cols, filled=True)

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


    @staticmethod
    def template_for_dec_tree_based_algo_GA(coin, s_date, lags_p_smas, lags_smas, date_split, model, num_folds,
                                            n_gens, pop_size, scoring, tourn_size, consec_stop, param_grid,
                                            lags_rsi=None, lags_std=None, search_term_=None, path_=None):
        dm = Data_manager(coin, s_date, search_term=search_term_,
                          path=path_)  # path='local_file' looks for files in local folder
        dm.download_price_data()
        dm.merge_search_with_price_data()
        dm.features_engineering_for_dec_tree(lags_p_smas, lags_smas, lags_rsi, lags_std)
        feature_cols = dm.feature_cols

        print('Feature cols:', feature_cols)

        ## Split data into training and test set
        data = dm.df
        training_data = data.loc[:date_split]
        test_data = data.loc[date_split:]

        ret_1 = dm.ret_1d
        X_train = training_data[feature_cols]
        Y_train = np.sign(training_data[ret_1])

        X_test = test_data[feature_cols]
        Y_test = np.sign(test_data[ret_1])

        ################################################## 1 ############################################################
        ##############################################################################################################
        is_dec_tree = False
        clf = None
        model_name = None

        if isinstance(model, DecisionTreeClassifier()):
            is_dec_tree = True
            model_name = 'Decision tree'
            clf = DecisionTreeClassifier()
        elif isinstance(model, RandomForestClassifier()):
            model_name = 'RF'
            clf = RandomForestClassifier()
        elif isinstance(model, GradientBoostingClassifier()):
            model_name = 'Grad Boost'
            clf = GradientBoostingClassifier()

        print('\n <--- Genetic search of features on ' + model_name + ' --->')
        evolved_model = GAFeatureSelectionCV(model,
                                            cv=num_folds,
                                            generations=n_gens,
                                            population_size=pop_size,
                                            scoring=scoring,
                                            tournament_size=tourn_size,
                                            keep_top_k=1,  #n of best solutions to keep
                                            verbose=True,
                                            n_jobs=-1)

        callback = ConsecutiveStopping(generations=consec_stop, metric='fitness')
        evolved_model.fit(X_train, Y_train, callbacks=callback)

        plot_fitness_evolution(evolved_model)

        best_features = list(X_train.columns[evolved_model.best_features_].values)
        print('Best features:', best_features)
        print('Best estimator:', evolved_model.best_estimator_)

        if is_dec_tree:
            text_representation = tree.export_text(model, feature_names=best_features) #list(best_features.values)
            print(text_representation)

        X_test_best_features = X_test.loc[:, best_features]
        X_train_best_features = X_train.loc[:, best_features]

        print('\nShape of X_test_best_features', X_test_best_features.shape)
        print('Shape of X_train_best_features', X_train_best_features.shape)

        # Predict only with the subset of selected features
        predictions = evolved_model.predict(X_test_best_features)
        print('\n- Accuracy score on test set (' + model_name + '):\t', accuracy_score(Y_test, predictions), '\n')
        Utils.show_confusion_matrix(Y_test, predictions, model_name + ' with best features found via ga')
        result_data = dm.get_result_data(test_data, predictions)
        Utils.plot_oos_results(result_data, 'Out of sample results using best features found via ga for ' + model_name)

        if is_dec_tree:
            plot_tree_(model, best_features)

        ################################################# 2 #############################################################
        ##############################################################################################################
        print('\n <--- Hyper params optimisation via genetic algo on '+ model_name +' --->')

        #TODO: change this and use a function as argument... porcata!!!!!!!!
        if is_dec_tree:
            param_grid.update({'max_features': Integer(2, len(best_features))})

        # The main class from sklearn-genetic-opt
        evolved_clf = GASearchCV(estimator=clf,
                                cv=num_folds,
                                scoring=scoring,
                                param_grid=param_grid,
                                population_size=pop_size,
                                generations=n_gens,
                                tournament_size=tourn_size,
                                n_jobs=-1,
                                keep_top_k=1,
                                verbose=True)

        callback = ConsecutiveStopping(generations=consec_stop, metric='fitness')
        evolved_clf.fit(X_train_best_features, Y_train, callbacks=callback)

        # Best parameters found
        print(evolved_clf.best_params_)
        # Use the model fitted with the best parameters
        predictions = evolved_clf.predict(X_test_best_features)
        print('- Accuracy score on test set Hyper params optimisation via genetic algo on '+
              model_name +':\t', accuracy_score(Y_test, predictions), '\n')
        Utils.show_confusion_matrix(Y_test, predictions, 'Hyper params optimisation via genetic algo on ' +
                                    model_name + '')
        result_data = dm.get_result_data(test_data, predictions)
        Utils.plot_oos_results(result_data, 'Out of sample results, hyper params optimisation via genetic algo on ' +
                               model_name + '')
