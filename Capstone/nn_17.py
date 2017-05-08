import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import TransformerMixin
import itertools
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint
from datetime import datetime

#Data has to be built into one large dataset
df_bats = pd.read_csv('battery_date_ward.csv')
df_crimdams = pd.read_csv('crimdams_date_ward.csv')
df_thefts = pd.read_csv('thefts_date_ward.csv')
df_narcotics = pd.read_csv('narcotics_date_ward.csv')
df_weather = pd.read_csv('weather_org.csv')
df_alots = pd.read_csv('abandoned_lots_date_ward.csv')

df_daterange = pd.DataFrame(pd.date_range(start='2010-01-02', end='2017-02-28', freq='D'),
	columns = ['Date'])

###Function that converts date column to datetime object and then reindex on datetime
def datetime_indexer(dataframe):
	dataframe['Date'] = dataframe['Unnamed: 0'].apply(lambda x: pd.to_datetime(x, format = '%Y-%m-%d'))
	dataframe.drop('Unnamed: 0', axis = 1, inplace = True)
	dataframe.set_index('Date', inplace = True)

datetime_indexer(df_bats)
y = df_bats[1:]

datetime_indexer(df_crimdams)
datetime_indexer(df_thefts)
datetime_indexer(df_narcotics)
datetime_indexer(df_weather)
datetime_indexer(df_alots)

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

df_bats = lag_func(df_bats)
df_crimdams = lag_func(df_crimdams)
df_thefts = lag_func(df_thefts)
df_narcotics = lag_func(df_narcotics)
df_weather = lag_func(df_weather)
df_alots = lag_func(df_alots)

df_daterange.set_index('Date', inplace = True)

df_bats.columns = [('Ward %s Batteries' % col) for col in df_bats.columns]
df_crimdams.columns = [('Ward %s CrimDamages' % col) for col in df_crimdams.columns]
df_thefts.columns = [('Ward %s Thefts' % col) for col in df_thefts.columns]
df_narcotics.columns = [('Ward %s Narcotics' % col) for col in df_narcotics.columns]
df_alots.columns = [('Ward %s AbandonedLots' % col) for col in df_alots.columns]

#So now we have assembled all of our data into one large pandas dataframe and we can begin pipelining each ward
df_whole = pd.concat([df_bats, df_crimdams, df_thefts, df_narcotics, df_weather, df_alots, df_daterange], axis = 1)

#So these pipes are ward specific
bats_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Batteries'), 
	StandardScaler())

crimdam_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 CrimDamages'), 
	StandardScaler())

thefts_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Thefts'),
	StandardScaler())

nar_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 Narcotics'),
	StandardScaler())

alots_pipe = make_pipeline(
	FeatureExtractor('Ward 17.0 AbandonedLots'),
	RobustScaler())

#These pipes are constants
max_temp_pipe = make_pipeline(
	FeatureExtractor('MaxTemp'),
	StandardScaler())

min_temp_pipe = make_pipeline(
	FeatureExtractor('MinTemp'),
	StandardScaler())

humid_pipe = make_pipeline(
	FeatureExtractor('Humidity'))

bar_pipe = make_pipeline(
	FeatureExtractor('BarPress'),
	StandardScaler())

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
	max_temp_pipe, min_temp_pipe, alots_pipe, bats_pipe, crimdam_pipe, thefts_pipe, nar_pipe, humid_pipe, bar_pipe, weekday_pipe, month_pipe, year_pipe, yearday_pipe)

x = union.fit_transform(df_whole)

#Defining y
def sep_func(row):
    if row < 5:
        return 0
    elif row > 8:
    	return 2
    else:
    	return 1

y_17 = y['17.0'].apply(lambda x: sep_func(x))

x_train, x_test, y_train, y_test = train_test_split(x, y_17, train_size = 0.66, random_state = 17)

def neural_net_opt(x, y, test_x, test_y):
	neural_net = MLPClassifier(solver = 'lbfgs', learning_rate = 'adaptive')
	a = [50, 100, 250]
	b = [1, 2, 3]
	grid = GridSearchCV(
		neural_net,
		param_grid = {
		'alpha': [0.000001, 0.00001, 0.0001, .001],
		'hidden_layer_sizes': [(x[0], x[1]) for x in list(itertools.product(a, b))],
		'activation': ['logistic', 'relu'],
		'learning_rate_init': [.001, .01, .1, 1, .0001]
		}, 
		cv= 3, verbose = 1, n_jobs = -1)
	grid.fit(x, y)
	print('Optimum params', grid.best_params_)
	neural_net_best = MLPClassifier(
		hidden_layer_sizes = grid.best_params_['hidden_layer_sizes'], 
		alpha = grid.best_params_['alpha'], 
		activation = grid.best_params_['activation'], 
		learning_rate_init = grid.best_params_['learning_rate_init'])
	neural_net_best.fit(x, y)
	predictions = neural_net_best.predict(test_x)
	confuse = pd.DataFrame(confusion_matrix(test_y, predictions))#, columns = ['Prediction = Norm', 'Prediction = Dangerous'], index = ['Actual = Norm', 'Actual = Dangerous'])
	print("Resulting Confusion Matrix\n" , confuse)
	print("Classification Report\n", classification_report(test_y, predictions))

neural_net_opt(x_train, y_train, x_test, y_test)

##After including a third class in our y variable

#increasing training size from .66 to .8
# Optimum params {'learning_rate_init': 0.01, 'alpha': 1e-06, 'activation': 'logistic', 'hidden_layer_sizes': (5, 2)}
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     274                      73
# Actual = Dangerous                130                      46
# Classification Report
#               precision    recall  f1-score   support

#           0       0.68      0.79      0.73       347
#           1       0.39      0.26      0.31       176

# avg / total       0.58      0.61      0.59       523

#change learning rate to adaptive
# Optimum params {'alpha': 1e-05, 'activation': 'logistic', 'hidden_layer_sizes': (7, 1), 'learning_rate_init': 0.01}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     402                     187
# Actual = Dangerous                135                     166
# Classification Report
#               precision    recall  f1-score   support

#           0       0.75      0.68      0.71       589
#           1       0.47      0.55      0.51       301

# avg / total       0.65      0.64      0.64       890

# # Optimum params {'hidden_layer_sizes': (3, 2), 'alpha': 1e-06, 'activation': 'logistic', 'solver': 'lbfgs'}
# Resulting Confusion Matrix
#                      Prediction = Norm  Prediction = Dangerous
# Actual = Norm                     412                     177
# Actual = Dangerous                157                     144
# Classification Report
#               precision    recall  f1-score   support

#           0       0.72      0.70      0.71       589
#           1       0.45      0.48      0.46       301

# avg / total       0.63      0.62      0.63       890