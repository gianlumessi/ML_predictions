from Utils import Utils
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn_genetic.space import Integer, Continuous, Categorical

coin = 'BTC-USD'
s_date = '2015-01-01'
date_split = '2021-01-01'
search_term = 'bitcoin'

# test options for classification
num_folds = 6
seed = 7
scoring = 'accuracy'

#For creating feature
lags_p_smas = [7, 14, 28, 60]
lags_smas = [7, 14, 28, 60]
lags_rsi = [7, 14, 28, 60]
lags_std = [7, 14, 28, 60]

# for GA optimisation
model = DecisionTreeClassifier(max_depth=4, criterion='gini', min_samples_leaf=10, splitter='best')
n_gens = 30
consec_stop = 5
pop_size = 800
tourn_size = 5
##'max_features': Integer(2, len(best_features))
param_grid = {'criterion': Categorical(['gini', 'entropy']),
              'max_depth': Integer(2, 4),
              'min_samples_leaf': Integer(2, 100)}


Utils.template_for_dec_tree_based_algo_GA(coin, s_date, lags_p_smas, lags_smas, date_split, model, num_folds,
                                            n_gens, pop_size, scoring, tourn_size, consec_stop, param_grid,
                                            lags_rsi=lags_rsi, lags_std=lags_std, search_term_=search_term,
                                          path_='local_file')

