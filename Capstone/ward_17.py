import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.base import TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import itertools

#Data has to be built into one large dataset
df_bats = pd.read_csv('battery_date_ward.csv')
df_crimdams = pd.read_csv('crimdams_date_ward.csv')
df_thefts = pd.read_csv('thefts_date_ward.csv')
df_narcotics = pd.read_csv('narcotics_date_ward.csv')
df_weather = pd.read_csv('weather_org.csv')
df_abandoned_lots = pd.read_csv('abandoned_lots_date_ward.csv')

df_daterange = pd.DataFrame(pd.date_range(start='2010-01-02', end='2017-02-28', freq='D'),
	columns = ['Date'])

###Function that converts date column to datetime object and then reindex on datetime
def datetime_indexer(dataframe):
	dataframe['Date'] = dataframe['Unnamed: 0'].apply(lambda x: pd.to_datetime(x, format = '%Y-%m-%d'))
	dataframe.drop('Unnamed: 0', axis = 1, inplace = True)
	dataframe.set_index('Date', inplace = True)

datetime_indexer(df_bats)
y = df_bats[1:]
# print(y[1:].sum().sort_values(ascending = False)) 

#Wards 28 24 17 20 6 have the most batteries
# for col in ['28.0', '24.0', '20.0', '6.0', '17.0']:
# 	print(y[col].describe()[6])

datetime_indexer(df_crimdams)
datetime_indexer(df_thefts)
datetime_indexer(df_narcotics)
datetime_indexer(df_weather)
datetime_indexer(df_abandoned_lots)

#Function for df_daterange that helps pull all pertinent datetime info
def date_pulls(dataframe):
    dataframe['Weekday'] = dataframe['Date'].apply(lambda x: x.weekday())
    dataframe['Month'] = dataframe['Date'].apply(lambda x: x.month)
    dataframe['Year'] = dataframe['Date'].apply(lambda x: x.year)
    dataframe['tt'] = dataframe['Date'].apply(lambda x: x.timetuple())
    dataframe['Yearday'] = dataframe['tt'].apply(lambda x: x.tm_yday)
date_pulls(df_daterange)

###Pulls the column that we want from dataframes
class FeatureExtractor(TransformerMixin):
    def __init__(self, column):
        self.column = column
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x, y=None):
        return x[self.column].values.reshape(-1, 1)

def lag_func(dataframe):
	return dataframe.shift(1)[1:]

#Function that converts negative ones to 0 (used in IsolationForest)
def zero_converter(value):
	if value == -1:
		return 1
	else:
		return 0

df_bats = lag_func(df_bats)
df_crimdams = lag_func(df_crimdams)
df_thefts = lag_func(df_thefts)
df_narcotics = lag_func(df_narcotics)
df_weather = lag_func(df_weather)
df_abandoned_lots = lag_func(df_abandoned_lots)

df_daterange.set_index('Date', inplace = True)

df_bats.columns = [('Ward %s Batteries' % col) for col in df_bats.columns]
df_crimdams.columns = [('Ward %s CrimDamages' % col) for col in df_crimdams.columns]
df_thefts.columns = [('Ward %s Thefts' % col) for col in df_thefts.columns]
df_narcotics.columns = [('Ward %s Narcotics' % col) for col in df_narcotics.columns]
df_abandoned_lots.columns = [('Ward %s Abandoned Lots' % col) for col in df_abandoned_lots.columns]
#So now we have assembled all of our data into one large pandas dataframe and we can begin pipelining each ward
df_whole = pd.concat([df_bats, df_crimdams, df_thefts, df_narcotics, df_weather, df_abandoned_lots, df_daterange], axis = 1)

#So these 4 pipes are the ones that our for loop should change at every iteration
bats_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Batteries'))

crimdam_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 CrimDamages'))

thefts_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Thefts'))

nar_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Narcotics'))

alots_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Abandoned Lots'))

#These pipes are constants
max_temp_pipe = make_pipeline(
	FeatureExtractor('MaxTemp'),
	MinMaxScaler((0,100)))

min_temp_pipe = make_pipeline(
	FeatureExtractor('MinTemp'), 
	MinMaxScaler((0, 100)))

humid_pipe = make_pipeline(
	FeatureExtractor('Humidity'))

bar_pipe = make_pipeline(
	FeatureExtractor('BarPress'))

weekday_pipe = make_pipeline(
	FeatureExtractor('Weekday'), 
	LabelBinarizer())

month_pipe = make_pipeline(
	FeatureExtractor('Month'),
	LabelBinarizer())

year_pipe = make_pipeline(
	FeatureExtractor('Year'),
	LabelBinarizer())

yearday_pipe = make_pipeline(
	FeatureExtractor('Yearday'),
	LabelBinarizer())

union = make_union(
	max_temp_pipe, min_temp_pipe, bats_pipe, crimdam_pipe, thefts_pipe, nar_pipe, alots_pipe, humid_pipe, bar_pipe, weekday_pipe, month_pipe, year_pipe, yearday_pipe)

x = union.fit_transform(df_whole)

#Defining y for a quick experiment
def sep_func(row):
    if row < 8:
        return 0
    else:
        return 1

y_17 = y['17.0'].apply(lambda x: sep_func(x))

x_train, x_test, y_train, y_test = train_test_split(x, y_17, train_size = 0.66, random_state = 17)

def randomforest_optimizer(x_tr, y_tr, test_x, test_y):
    rfc = RandomForestClassifier()
    grid = GridSearchCV(
      rfc, 
      param_grid = {'n_estimators': [10, 50, 100, 250], 
                    'criterion': ['gini', 'entropy'], 
                    'max_features': ['auto', 'sqrt', 'log2'], 
                    'bootstrap': [True, False],
                    'warm_start': [True, False]}, 
      verbose = 1, cv = 5, scoring = 'f1')
    grid.fit(x_tr, y_tr)
    print('\n******** RANDOM FOREST ********\n')
    print("Optimum parameters: ", grid.best_params_)
    rfc_best_fit = RandomForestClassifier(
      n_estimators = grid.best_params_['n_estimators'], 
      criterion = grid.best_params_['criterion'], 
      max_features = grid.best_params_['max_features'],
      bootstrap = grid.best_params_['bootstrap'], 
      warm_start = grid.best_params_['warm_start'])
    rfc_best_fit.fit(x_tr, y_tr)
    # f_importance = []
    # for items in zip(x_tr.columns, rfc_best_fit.feature_importances_):
    #     f_importance.append(items)
    # print('Feature importances\n', sorted(f_importance, key = lambda x: x[1], reverse = True))
    predictions = rfc_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

# rfc_best_params = randomforest_optimizer(x_train, y_train, x_test, y_test)

ss = StandardScaler()
x_train_s = ss.fit_transform(x_train)
x_test_s = ss.transform(x_test)

def knn_optimizer(x, y, test_x, test_y):
    knn = KNeighborsClassifier()
    grid = GridSearchCV(
      knn, 
      param_grid = {
                  'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10], 
                  'weights': ['uniform', 'distance']
                  }, 
      verbose = 1, scoring = 'f1'
      )
    grid.fit(x, y)
    print('\n******** K NEAREST NEIGHBORS ********\n')
    print("Optimum parameters: ", grid.best_params_)
    knn_best_fit = KNeighborsClassifier(n_neighbors = grid.best_params_['n_neighbors'], weights = grid.best_params_['weights'])
    knn_best_fit.fit(x, y)
    predictions = knn_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

# knn_best_params = knn_optimizer(x_train_s, y_train, x_test_s, y_test)

def gradientBoosted_optimizer(x, y, test_x, test_y):
    gbc = GradientBoostingClassifier()
    grid = GridSearchCV(
      gbc, 
      param_grid = {'n_estimators': [100, 500, 1000],
                    'max_features': ['auto', 'sqrt', 'log2'],
                    'warm_start': [True, False]}, 
      verbose = 1, scoring = 'f1', cv = 5)
    grid.fit(x, y)
    print('\n******** GRADIENT BOOSTING ********\n')
    print("Optimum parameters: ", grid.best_params_)
    gbc_best_fit = GradientBoostingClassifier(
      n_estimators = grid.best_params_['n_estimators'], 
      max_features = grid.best_params_['max_features'],
      warm_start = grid.best_params_['warm_start'])
    gbc_best_fit.fit(x, y)
    # f_importance = []
    # for items in zip(x.columns, gbc_best_fit.feature_importances_):
    #     f_importance.append(items)
    # print('Feature importances\n', sorted(f_importance, key = lambda x: x[1], reverse = True))
    predictions = gbc_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

# gbc_best_params = gradientBoosted_optimizer(x_train, y_train, x_test, y_test)

def adaboost_optimizer(x, y, test_x, test_y):
    abc = AdaBoostClassifier()
    grid = GridSearchCV(
      abc, 
      param_grid = {'n_estimators': [50, 100, 250, 500],
                    'algorithm': ['SAMME', 'SAMME.R']}, 
      verbose = 1, scoring = 'f1', cv = 5)
    grid.fit(x, y)
    print('\n******** ADABOOSTING ********\n')
    print("Optimum parameters: ", grid.best_params_)
    abc_best_fit = AdaBoostClassifier(
      n_estimators = grid.best_params_['n_estimators'], 
      algorithm = grid.best_params_['algorithm'])
    abc_best_fit.fit(x, y)
    predictions = abc_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

# abc_best_params = adaboost_optimizer(x_train, y_train, x_test, y_test)

def isoforest_optimizer(x_tr, y_tr, test_x, test_y):
    iso = IsolationForest()
    grid = GridSearchCV(
      iso,
      param_grid = {'n_estimators': [10, 50, 100, 250],
                    'contamination': [0.05, 0.1, 0.15, 0.2],
                    'bootstrap': [True, False]}, 
      verbose = 1, scoring = 'roc_auc')
    grid.fit(x_tr, y_tr)
    print("Highest recall score: ", grid.best_score_)
    print("The parameters that yield the greatest score: ", grid.best_params_)
    iso_best_fit = IsolationForest(
      n_estimators = grid.best_params_['n_estimators'], 
      bootstrap = grid.best_params_['bootstrap'], 
      contamination = grid.best_params_['contamination'])
    iso_best_fit.fit(x_tr, y_tr)
    predictions = iso_best_fit.predict(test_x)
    predictions = [zero_converter(values) for values in predictions]
    confuse = confusion_matrix(test_y, predictions)
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))

# isoforest_optimizer(x_train, y_train, x_test, y_test)

def mnb_optimizer(x, y, test_x, test_y):
    mnb = MultinomialNB()
    grid = GridSearchCV(
      mnb, 
      param_grid = {'alpha': [0, 0.25, 0.5, 0.75, 1], 
                    'fit_prior': [True, False]}, 
      verbose = 1, scoring = 'f1', cv = 5)
    grid.fit(x, y)
    print('\n****** Multinomial Naive Bayes ********\n')
    print("Optimum parameters: ", grid.best_params_)
    mnb_best_fit = MultinomialNB(
        alpha = grid.best_params_['alpha'],
        fit_prior = grid.best_params_['fit_prior'])
    mnb_best_fit.fit(x, y)
    predictions = mnb_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

mnb_best_params = mnb_optimizer(x_train, y_train, x_test, y_test)

def logreg_optimizer(x, y, test_x, test_y):
    logit = LogisticRegression()
    grid = GridSearchCV(
      logit, 
      param_grid = {'penalty': ['l1', 'l2'], 
                    'C': [.001, .01, .1, 1, 10, 100], 
                    'fit_intercept': [True, False], 
                    'tol': [.0001, .001, .01, .1, 1, 10]}, 
      verbose = 1, scoring = 'f1', cv = 5)
    grid.fit(x, y)
    print('\n******** LOGISTIC REGRESSION ********\n')
    print("Optimum parameters: ", grid.best_params_)
    logit_best_fit = LogisticRegression(
      penalty = grid.best_params_['penalty'], 
      C = grid.best_params_['C'], 
      fit_intercept = grid.best_params_['fit_intercept'],
      tol = grid.best_params_['tol'])
    logit_best_fit.fit(x, y)
    predictions = logit_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

logreg_best_params = logreg_optimizer(x_train, y_train, x_test, y_test) #Thus far, logreg has been the best at predicting 1s

def extratrees_optimizer(x_tr, y_tr, test_x, test_y):
    xtc = ExtraTreesClassifier()
    grid = GridSearchCV(
      xtc, 
      param_grid = {'n_estimators': [10, 50, 100, 250], 
                    'criterion': ['gini', 'entropy'], 
                    'max_features': ['auto', 'sqrt', 'log2'], 
                    'bootstrap': [True, False],
                    'warm_start': [True, False]}, 
      verbose = 1, scoring = 'f1', cv = 5)
    grid.fit(x_tr, y_tr)
    print('\n******** EXTRA TREES ********\n')
    print("Optimum parameters: ", grid.best_params_)
    xtc_best_fit = ExtraTreesClassifier(
      n_estimators = grid.best_params_['n_estimators'], 
      criterion = grid.best_params_['criterion'], 
      max_features = grid.best_params_['max_features'],
      bootstrap = grid.best_params_['bootstrap'], 
      warm_start = grid.best_params_['warm_start'])
    xtc_best_fit.fit(x_tr, y_tr)
    # f_importance = []
    # for items in zip(x_tr.columns, rfc_best_fit.feature_importances_):
    #     f_importance.append(items)
    # print('Feature importances\n', sorted(f_importance, key = lambda x: x[1], reverse = True))
    predictions = xtc_best_fit.predict(test_x)
    confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
    print("Resulting Confusion Matrix\n" , confuse)
    print("Classification Report\n", classification_report(test_y, predictions))
    return grid.best_params_

xtc_best_params = extratrees_optimizer(x_train, y_train, x_test, y_test)

# rfc = RandomForestClassifier(bootstrap = rfc_best_params['bootstrap'], max_features = rfc_best_params['max_features'], warm_start = rfc_best_params['warm_start'], n_estimators = rfc_best_params['n_estimators'], criterion = rfc_best_params['criterion'])
# knn = KNeighborsClassifier(n_neighbors = knn_best_params['n_neighbors'], weights = knn_best_params['weights'])
logit = LogisticRegression(tol = logreg_best_params['tol'], C = logreg_best_params['C'], fit_intercept = logreg_best_params['C'], penalty = logreg_best_params['penalty'])
# gbc = GradientBoostingClassifier(n_estimators = gbc_best_params['n_estimators'], max_features = gbc_best_params['max_features'], warm_start = gbc_best_params['warm_start'])
# abc = AdaBoostClassifier(n_estimators = 500, algorithm = 'SAMME')
mnb = MultinomialNB(alpha = mnb_best_params['alpha'], fit_prior = mnb_best_params['fit_prior'])
xtc = ExtraTreesClassifier(n_estimators = xtc_best_params['n_estimators'], max_features = xtc_best_params['max_features'], warm_start = xtc_best_params['warm_start'], criterion = xtc_best_params['criterion'], bootstrap = xtc_best_params['bootstrap'])

def vote_optimizer(x, y, test_x, test_y):
	vc = VotingClassifier(estimators = [('logit', logit), ('xtc', xtc), ('mnb', mnb)])
	a = b = c = [1, 2, 3]
	grid = GridSearchCV(
		vc, scoring = 'f1',
		param_grid = {
		'weights': [(x[0], x[1], x[2]) for x in list(itertools.product(a, b, c))],
		'voting': ['hard']})
	grid.fit(x, y)
	print('\n******* VOTING CLASSIFIER *******\n')
	print("Optimum parameters: ", grid.best_params_)
	vc_best_fit = VotingClassifier(
	  estimators = [('logit', logit), ('xtc', xtc), ('mnb', mnb)],
      weights = grid.best_params_['weights'], 
      voting = grid.best_params_['voting'])
	vc_best_fit.fit(x, y)
	predictions = vc_best_fit.predict(test_x)
	confuse = pd.DataFrame(confusion_matrix(test_y, predictions), columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
	print("Resulting Confusion Matrix\n" , confuse)
	print("Classification Report\n", classification_report(test_y, predictions))


vote_optimizer(x_train, y_train, x_test, y_test)

#voting class results
# Optimum parameters:  {'voting': 'hard', 'weights': (1, 1, 3)}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     336                     253
# Actual = Dangerous                111                     190
# Classification Report
#               precision    recall  f1-score   support

#           0       0.75      0.57      0.65       589
#           1       0.43      0.63      0.51       301

# avg / total       0.64      0.59      0.60       890

# ******** LOGISTIC REGRESSION ********

# Optimum parameters:  {'fit_intercept': True, 'C': 10, 'tol': 0.0001, 'penalty': 'l2'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     457                     132
# Actual = Dangerous                174                     127
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.78      0.75       589
#           1       0.49      0.42      0.45       301

# avg / total       0.65      0.66      0.65       890

# Fitting 5 folds for each of 96 candidates, totalling 480 fits
# [Parallel(n_jobs=1)]: Done 480 out of 480 | elapsed:  4.3min finished

# ******** EXTRA TREES ********

# Optimum parameters:  {'bootstrap': False, 'warm_start': True, 'n_estimators': 100, 'criterion': 'entropy', 'max_features': 'sqrt'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     522                      67
# Actual = Dangerous                210                      91
# Classification Report
#               precision    recall  f1-score   support

#           0       0.71      0.89      0.79       589
#           1       0.58      0.30      0.40       301

# avg / total       0.67      0.69      0.66       890

# ****** Multinomial Naive Bayes ********

# Optimum parameters:  {'fit_prior': False, 'alpha': 0.25}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     336                     253
# Actual = Dangerous                111                     190
# Classification Report
#               precision    recall  f1-score   support

#           0       0.75      0.57      0.65       589
#           1       0.43      0.63      0.51       301

# avg / total       0.64      0.59      0.60       890

# ******** RANDOM FOREST ********

# Optimum parameters:  {'bootstrap': False, 'warm_start': False, 'n_estimators': 50, 'criterion': 'entropy', 'max_features': 'auto'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     523                      66
# Actual = Dangerous                229                      72
# Classification Report
#               precision    recall  f1-score   support

#           0       0.70      0.89      0.78       589
#           1       0.52      0.24      0.33       301

# avg / total       0.64      0.67      0.63       890

# Fitting 3 folds for each of 16 candidates, totalling 48 fits
# [Parallel(n_jobs=1)]: Done  48 out of  48 | elapsed:  1.0min finished

# ******** K NEAREST NEIGHBORS ********

# Optimum parameters:  {'n_neighbors': 5, 'weights': 'uniform'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     443                     146
# Actual = Dangerous                203                      98
# Classification Report
#               precision    recall  f1-score   support

#           0       0.69      0.75      0.72       589
#           1       0.40      0.33      0.36       301

# avg / total       0.59      0.61      0.60       890

# Fitting 5 folds for each of 18 candidates, totalling 90 fits
# [Parallel(n_jobs=1)]: Done  90 out of  90 | elapsed:  4.5min finished

# ******** GRADIENT BOOSTING ********

# Optimum parameters:  {'warm_start': False, 'n_estimators': 500, 'max_features': 'sqrt'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     503                      86
# Actual = Dangerous                201                     100
# Classification Report
#               precision    recall  f1-score   support

#           0       0.71      0.85      0.78       589
#           1       0.54      0.33      0.41       301

# avg / total       0.65      0.68      0.65       890

# Fitting 5 folds for each of 8 candidates, totalling 40 fits
# [Parallel(n_jobs=1)]: Done  40 out of  40 | elapsed:  1.2min finished

# ******** ADABOOSTING ********

# Optimum parameters:  {'n_estimators': 100, 'algorithm': 'SAMME.R'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     487                     102
# Actual = Dangerous                189                     112
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.83      0.77       589
#           1       0.52      0.37      0.43       301

# avg / total       0.65      0.67      0.66       890


#


#Setting cut off line to 10
# Optimum parameters:  {'voting': 'hard', 'weights': (1, 3, 3, 1, 3)}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     737                      15
# Actual = Dangerous                127                      11
# Classification Report
#               precision    recall  f1-score   support

#           0       0.85      0.98      0.91       752
#           1       0.42      0.08      0.13       138

# avg / total       0.79      0.84      0.79       890

#Setting the cut off line to 9
# Optimum parameters:  {'weights': (1, 3, 3, 1, 3), 'voting': 'hard'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     622                      59
# Actual = Dangerous                167                      42
# Classification Report
#               precision    recall  f1-score   support

#           0       0.79      0.91      0.85       681
#           1       0.42      0.20      0.27       209

# avg / total       0.70      0.75      0.71       890

#After making below updates the results are
# Optimum parameters:  {'voting': 'hard', 'weights': (1, 3, 3, 1, 3)}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     478                     111
# Actual = Dangerous                188                     113
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.81      0.76       589
#           1       0.50      0.38      0.43       301

# avg / total       0.65      0.66      0.65       890

#VC results when scoring set to recall for all classifiers
#Adaboost over GradientBoost (drop xgb)
#LogReg still doing great
#KNN suffered badly when set to recall
#RF did too so maybe keep those as accuracy score
#Xtra trees barely underperformed Logreg so it is droppable 

# ******* VOTING CLASSIFIER *******

# Optimum parameters:  {'voting': 'hard', 'weights': (1, 1, 3, 1, 1, 3, 1)}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     498                      91
# Actual = Dangerous                186                     115
# Classification Report
#               precision    recall  f1-score   support

#           0       0.73      0.85      0.78       589
#           1       0.56      0.38      0.45       301

# avg / total       0.67      0.69      0.67       890

#After making the above changes we see our results become



#WINNER THUS FAR
#gbc, logit, rfc
#weights = 1,2,1 yielded and bad day cut off = 9
# Resulting Confusion Matrix
#  [[527  62]
#  [206  95]]
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.89      0.80       589
#           1       0.61      0.32      0.41       301

# avg / total       0.68      0.70      0.67       890



##TIED
#weights = 1,5,1 yielded
# Resulting Confusion Matrix
#  [[526  63]
#  [205  96]]
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.89      0.80       589
#           1       0.60      0.32      0.42       301

# avg / total       0.68      0.70      0.67       890

