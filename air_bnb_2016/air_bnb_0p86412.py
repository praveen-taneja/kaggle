# Aim - Predict which countries are users likely to book.
# 
# Other than user and session data, we also have some data about different 
# countries and their demographics. Some of the data is not available for test 
# file, so can't be directly used as features for training.

#%%

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import sklearn.linear_model as linear_model
import sklearn.ensemble as ensemble
import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics
import sklearn.grid_search as grid_search
import sklearn.preprocessing as preprocessing
import os

import os
os.chdir("/Users/taneja/Copy/machine_learning/abb")
#%%
def ndcg(y, y_pred): # metrics.make_scorer(score_func)
    # return accuracy score
    pass
    
#%%
#gbc = ensemble.GradientBoostingClassifier(n_estimators=50, verbose = 1)
#pars = {'learning_rate': [0.01, 0.05, 0.1]}
#gbc = ensemble.GradientBoostingClassifier(n_estimators=50, verbose = 1)
#pars = {'learning_rate': [0.01, 0.05, 0.1]}
#clf = grid_search.GridSearchCV(gbc, pars)
clf = ensemble.GradientBoostingClassifier(n_estimators=50, learning_rate = 0.05,
                                          verbose = 1)
#%%
countries = pd.read_table('./data/countries.csv', delimiter = ',')
age_gender = pd.read_table('./data/age_gender_bkts.csv', delimiter = ',')
test_users  = pd.read_table('./data/test_users.csv', delimiter = ',')
sessions = pd.read_table('./data/sessions.csv', delimiter = ',')
sample_submission = pd.read_table('./data/sample_submission_NDF.csv', 
                                  delimiter = ',')
train_users_2 = pd.read_table('./data/train_users_2.csv', delimiter = ',')
train_nrows = train_users_2.shape[0]
train_test = pd.concat([train_users_2, test_users], axis = 0)
# date_first_booking is not present in test data.
train_test = train_test.drop(['date_first_booking'], axis = 1)

#%%
print train_test.describe()
#%%
print countries.describe()
#%%
print age_gender.describe()
#%%
print sessions.describe()
#%%
#Let's start with just the train_users_2 to make initial prediction.
train_drop_cols = []

categorical = []
#categorical.append('id')
categorical.append('gender')
categorical.append('signup_method')
categorical.append('signup_flow')
categorical.append('language')
categorical.append('affiliate_channel')
categorical.append('affiliate_provider')
categorical.append('first_affiliate_tracked')
categorical.append('signup_app')
categorical.append('first_device_type')
categorical.append('first_browser')

print categorical

train_test = pd.get_dummies(train_test, columns = categorical)

#%%
'''
Numerical features:
1. date_account_created
2. timestamp_first_active
3. age has some bad values (corrupted by year of birth)
'''

#%% extract year, month, date, day, hour

train_test['date_account_created'] = pd.to_datetime(train_test['date_account_created'])
train_test['day_date_account_created'] = [t.dayofweek for t in train_test['date_account_created']]
train_test['date_date_account_created'] = [t.day for t in train_test['date_account_created']]
train_test['month_date_account_created'] = [t.month for t in train_test['date_account_created']]
train_test['year_date_account_created'] = [t.year for t in train_test['date_account_created']]
print train_test['day_date_account_created'].head()
print train_test['date_date_account_created'].head()
print train_test['month_date_account_created'].head()
print train_test['year_date_account_created'].head()

#%% extract year, month, date, day, hour
ts_covert = lambda x: pd.datetime.strptime(str(x), '%Y%m%d%H%M%S')
timestamp_first_active_tmp = train_test['timestamp_first_active'].map(ts_covert)
print timestamp_first_active_tmp.head()
train_test['year_timestamp_first_active'] = timestamp_first_active_tmp.map(lambda x: x.year)
train_test['month_timestamp_first_active'] = timestamp_first_active_tmp.map(lambda x: x.month)
train_test['date_timestamp_first_active'] = timestamp_first_active_tmp.map(lambda x: x.day)
train_test['day_timestamp_first_active'] = timestamp_first_active_tmp.map(lambda x: x.dayofweek)
train_test['hour_timestamp_first_active'] = timestamp_first_active_tmp.map(lambda x: x.hour)
print train_test['year_timestamp_first_active'].head()
print train_test['month_timestamp_first_active'].head()
print train_test['date_timestamp_first_active'].head()
print train_test['day_timestamp_first_active'].head()
print train_test['hour_timestamp_first_active'].head()

#%%
print train_test.columns.values
train_test_drop_cols = ['id', 'date_account_created', 'timestamp_first_active']
train_test = train_test.drop(train_test_drop_cols, axis = 1)
#train = train.drop(train.columns.values)

#%% impute bad values with mode
print train_test['age'].describe()
criteria = (train_test['age'] < 18.0) | (train_test['age'] > 110.0) | (train_test['age'].isnull())
train_test['age'].loc[criteria] = train_test['age'].mode()[0]
print train_test['age'].iloc[193]

#%% split train_test into train and test. keep as data frames for now
y_train = train_test['country_destination'][:train_nrows]
train_test = train_test.drop(['country_destination'], axis = 1)
X_train = train_test.iloc[:train_nrows, :]
X_test = train_test.iloc[train_nrows:, :]

#%% split training data into training and testing
le = preprocessing.LabelEncoder()
sorted_classes = np.sort(y_train.unique())
print sorted_classes
#y_train_encd = le.fit_transform(y_train.values)
#%%
X_train, X_train_test, y_train, y_train_test = cross_validation.train_test_split(
                                X_train.values, y_train.values,test_size = 0.1,
                                random_state=42)
                                
#%% fit and predict
clf.fit(X_train, y_train)
print 'fitting over'
#%%
##########print 'best learning rate =', clf.best_params_
#%%
y_train_test_predicted = clf.predict(X_train_test)
print 'accuracy =', metrics.accuracy_score(y_train_test, y_train_test_predicted)
#%%
#predictions['actual'].value_counts().plot(kind = 'bar', color = 'red')
#predictions['predicted'].value_counts().plot(kind = 'bar', color = 'blue')
#%% make predictions on actual test data
test_predicted = clf.predict_proba(X_test.values)
test_predicted = pd.DataFrame(test_predicted, columns = sorted_classes)
users_id = pd.DataFrame({'id':test_users['id']})
predictions_wide = pd.concat([users_id, test_predicted], axis = 1)
#%%
predictions_wide_T = predictions_wide.transpose()
#col = predictions_wide_T[0]
#col_srt = col.sort_values(ascending = False)
#print col_srt.index.values[1:6]
#print col['id']
#d = {'id':[], 'country':[]}
#%%
ids = []
countries = []
for col in predictions_wide_T:
    col = predictions_wide_T[col]
    #print col
    ids.extend([col['id']]*5)
    col_srt = col.sort_values(ascending = False)
    countries.extend(col_srt.index.values[1:6])

predictions_out = pd.DataFrame({'id': ids, 'country' : countries})
predictions_out = predictions_out[['id', 'country']]
predictions_out.to_csv('./data/predictions_01_17_16.csv', index = False)

#    col.order()
#for ix, row in predictions_wide.iterrow():
#    user_id = row[0]
#    countries = row[1:]
    
#%%

#predictions = pd.DataFrame({'id':test_users['id'], 'country':test_predicted})
#predictions = predictions[['id', 'country']]
#predictions.to_csv('./data/predictions_01_14_16.csv', index = False)
