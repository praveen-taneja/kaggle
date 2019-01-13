# Walmart trip type classification: Based on the day of the week and items 
# purchase, we need to predict TripType.

# Author: Praveen Taneja
#%%

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')
import collections
import sklearn.metrics as metrics
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear_model
import sklearn.cross_validation as cross_validation
import sklearn.grid_search as grid_search
import sklearn.pipeline as pipeline
import scipy.stats as stats
import os

os.chdir("/Users/taneja/OneDrive/from_copy/machine_learning/wmt")
#os.chdir("C:/Users/praveen.taneja/Copy/machine_learning/wmt")
#%%

train = pd.read_table('./data/train.csv', delimiter = ',')
test = pd.read_table('./data/test.csv', delimiter = ',')
sample_submission = pd.read_table('./data/sample_submission.csv', 
                                  delimiter = ',')

print train.head(2)
print train.tail(2)
print ' '
print ' '
print test.head(2)
print test.tail(2)
print ' '
print ' '
print sample_submission.head(2)
print sample_submission.tail(2)

#%%
train_n_rows = train.shape[0]
# combine because many operations can be done together on both datasets
unique_days = train['Weekday'].unique() # for later use
train_test = pd.concat([train, test], axis = 0)
#%%
# missing values
print 'missing values in different cols'
print ' '
print train_test.isnull().sum()
print 'train_test.shape =', train_test.shape
print '% missing values in Upc, FinelineNumber =', (8115.0/1300700.0)*100
print '% missing values in DepartmentDescription =', (2689.0/1300700.0)*100

#%%
# missing values
'''
We can't drop rows with missing values even though % missing is quite small, 
as we are required to predict outcome for all visits in test data.
We could do one of the following.

1. Set them to a distinct new category or value. Eg. DepartmentDescription = 
No_DepartmentDescription.
UPC = No_UPC; FinelineNumber = No_FinelineNumber. Makes least assumptions.
2. Set them equal to most probable value for that category. 
3. More complicated imputations.

For now lets go with option 1 as there are not so many missing values.
'''
train_test['Upc'] = train_test['Upc'].fillna('Missing_Upc')
train_test['FinelineNumber'] = (train_test['FinelineNumber'].
                            fillna('Missing_FinelineNumber'))
train_test['DepartmentDescription'] = (train_test['DepartmentDescription'].
                            fillna('Missing_DepartmentDescription'))

test['Upc'] = test['Upc'].fillna('Missing_Upc')
test['FinelineNumber'] = (test['FinelineNumber'].
                            fillna('Missing_FinelineNumber'))
test['DepartmentDescription'] = (test['DepartmentDescription'].
                            fillna('Missing_DepartmentDescription'))

#%%

print len(train_test['TripType'].unique())
print len(train_test['VisitNumber'].unique())
print len(train_test['Weekday'].unique())
print len(train_test['Upc'].unique())
print len(train_test['ScanCount'].unique())
print len(train_test['DepartmentDescription'].unique())
print len(train_test['FinelineNumber'].unique())

#There are total of 1300700 rows in train_test, one for each item.  
#But only 191348 unique visit numbers. This is because many different rows have 
#same visit number. Later we will convert data to wide format with one row per 
#visit because we need to predict trip type for each visit
#%%

def to_wide(data, groupby_key):
    ''' The data has many rows that have same visit number and trip type. 
    These correspond to different items purchased. Column names are as follows.
    
    TripType, VisitNumber, Weekday, Upc, ScanCount, DepartmentDescription, 
    FinelineNumber
    
    We want each visit to be represented in a single row. TripType, 
    VisitNumber, Weekday are same for all items in a single visit. For Upc 
    which is item code we can have columns for all unique UPCs and each row 
    entry can be sum of ScanCount for that UPC. Similarly, we
    can have different columns for different FinelineNumber and for each row 
    the entry can sum of FinelineNumber for that UPC.
    
    Implementation - Convert to pandas data frame. Group by visit number. We 
    can make the dataframe in the begining and fill it row by row.
    '''
    
    unique_visits = data['VisitNumber'].unique()
    unique_days = data['Weekday'].unique()
    #unique_upcs = data['Upc'].unique()
    unique_dept_desc = data['DepartmentDescription'].unique()
    #unique_fine_line_num = data['FinelineNumber'].unique()
    
    # additional columns
    additional_columns = ['unique_purchases', 'unique_returns',
                          'total_purchases']
    
    num_unique_visits = len(unique_visits)
    print 'num_unique_visits =', num_unique_visits
    
    cols = ['TripType', 'VisitNumber']
    for day in unique_days:
        cols.append(day)
    for dept_desc in unique_dept_desc:
        cols.append(str(dept_desc))
    #for upc in unique_upcs:
    #    cols.append(upc)
    #for fine_line_num in unique_fine_line_num:
    #    cols.append(str(fine_line_num))
    for additional_column in additional_columns:
        cols.append(str(additional_column))
    
    
    # initialize
    d = collections.OrderedDict()
    for col in cols:
        d[col] = [0]*num_unique_visits
        
    grouped = data.groupby(groupby_key)
    #print 'groupby_key', groupby_key
    
    i = 0    
    for name, group in grouped:
                
        d['TripType'][i] =  group['TripType'].iloc[0]
        d['VisitNumber'][i] =  group['VisitNumber'].iloc[0]
        
        day = group['Weekday'].iloc[0]
        d[day][i] =  1
        
        depts = group['DepartmentDescription'].unique()
        for dept in depts:
            
            d[str(dept)][i] = (group['DepartmentDescription']
                                [group['DepartmentDescription'] == dept].
                                count())
        
        '''
        
        fine_line_nums = group['FinelineNumber'].unique()
        for fine_line_num in fine_line_nums:
            d[str(fine_line_num)][i] = 
            group['ScanCount'][group['FinelineNumber'] == fine_line_num].count()
        
        '''
        upcs = group['Upc'][group['ScanCount'] >= 0].unique()
        d['unique_purchases'][i] = len(upcs)
        
        upcs_returned = group['Upc'][group['ScanCount'] < 0].unique()
        d['unique_returns'][i] = len(upcs_returned)
        
        upcs = group['ScanCount'][group['ScanCount'] >= 0]
        d['total_purchases'][i] = upcs.sum()
        
            
        i = i + 1
 
    return pd.DataFrame(d)


#%%

train_test_wide =  to_wide(train_test, 'VisitNumber')
# save to disk so we don't have to create it again the next time
train_test_wide.to_csv('./data/train_test_wide.csv', index = False)
#%%
# load from disk, if needed
train_test_wide = pd.read_table('./data/train_test_wide.csv', delimiter = ',')
#%%
train = train_test_wide[train_test_wide['TripType'].notnull()]
test = train_test_wide[train_test_wide['TripType'].isnull()]
print sorted(train_test_wide['TripType'].unique())

#%%
#Exploratory data analysis

#1. Class frequency - Which trip types are more common than others? 

fig = plt.figure(figsize = (8,6))
fig = train_test_wide['TripType'].value_counts().plot(kind = 'bar')
plt.xlabel('TripType')
plt.ylabel('Counts')
print train_test_wide['TripType'].value_counts()
'''
TripType8 is most common, TripType14 is least (has only 4 visits!)
'''
#%%
#2. Does TripType999 corrospond to returns? 
grouped = train.groupby(['TripType'])

for name, group in grouped:
    #plt.figure()
    color = 'black'
    if name == 999:
        color = 'red'
    vals, binEdges=np.histogram(group['unique_returns'], bins=5, range = [0, 5])
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plt.plot(bincenters,vals,'-', color = color)
    plt.xlim([0, 5])
    #group['unique_returns'].hist(range = [0, 5], normed = 'True', label = name)#value_counts().plot
    #plt.legend()
plt.xlabel('unique_returns')
plt.ylabel('Counts')

'''
Compared to other TripTypes, 999 has most unique returns (ScanCount < 0) .
Although in many cases TripTypes = 999, even when ScanCount >0 for all purchases)
Eg. 999	207	Friday	76163520390	1	PRODUCE
    999	207	Friday	2242229000	2	DSD GROCERY
    
    999	295	Friday	1019906311	1	OFFICE SUPPLIES
    
    999	351	Friday	68113178251	1	FINANCIAL SERVICES
    999	351	Friday	68113178252	1	FINANCIAL SERVICES
    999	357	Friday	60538807733	1	FINANCIAL SERVICES
    999	357	Friday	68113107941	1	FINANCIAL SERVICES
'''
#%%
#2. How do trip types vary by unique purchases?
grouped = train.groupby(['TripType'])
color = 'black'
for name, group in grouped:
    color = 'black'
    if name == 39: # 2nd most popular trip type
        color = 'red' 
    if name == 3:
        color = 'blue'
    if name == 8: # most popular trip type
        color = 'green'
    if name == 999: # TripTypes with most returns
        color = 'purple'
    vals, binEdges=np.histogram(group['unique_purchases'], bins=10, range = [0, 10])
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    print 'TripType', name
    print vals
    plt.plot(bincenters,vals,'-', color = color)
    plt.xlim([0, 10])
    #group['unique_returns'].hist(range = [0, 5], normed = 'True', label = name)#value_counts().plot
    #plt.legend()
plt.xlabel('unique_purchases')
plt.ylabel('Counts')
    
'''
While most trips have 1 unique purchase, TripType3 has 2 unique most often.
TripType39 has biggest number of unique purchases. TripType39 is also the
2nd most popular TripType. Likely TripType39 is weekly shopping.
TripType999 (unlike other trips) has 0 as the most unique purchase.

'''
#%%
#3. Trip type should depend on the day of the week. Weekly grocery shopping
# usually happens on the weekend.

for day in unique_days:
    fig = plt.figure()
    day_data = train[train[day] == 1]
    day_data['TripType'].value_counts().plot(kind = 'bar', label = day)
    plt.xlabel('TripType')
    plt.ylabel('Counts')    
    plt.legend()

'''
Trip types don't seem to depend that much on days of the week. For example, 
Trip types 8, 39, 9, 999 are in top 5 every day. Though the overall number of
visits are higher on Friday, Saturday, Sunday.
'''
#%%
# Are 'unique_purchases' and 'total_purchases' correlated? If so, we shouldn't 
# include both
train.plot('unique_purchases', 'total_purchases', kind = 'scatter')
plt.xlim((0, 150))
plt.ylim((0, 150))
y1 = train['unique_purchases'].values
y2 = train['total_purchases'].values
print 'Correlation between unique and total purchases =', stats.pearsonr(y1, y2)

# Dropping total_purchases as it's highly correlated to unique_purchases
train = train.drop('total_purchases', axis = 1)
test = test.drop('total_purchases', axis = 1)

#%%
# Feature selection

y_train = train['TripType'].values
X_train = train.drop(['TripType', 'VisitNumber'], axis = 1).values
X_test = test.drop(['TripType', 'VisitNumber'], axis = 1).values

clf = ensemble.RandomForestClassifier(n_estimators=50)
'''
clf.fit(X_train, y_train)
important_features = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis = 0)

indices = np.argsort(important_features)[::-1] # sort in descending order
col_names = (train.drop(['TripType', 'VisitNumber'], axis = 1).
                        columns.values)

for col in range(X_train.shape[1]):
    print (col, indices[col], col_names[indices[col]], 
    important_features[indices[col]])
plt.bar(range(X_train.shape[1]), important_features[indices], 
        yerr= std[indices], color = 'red')

top_features = col_names[indices][0:40]
top_features_w_TripType = ['TripType'] + list(top_features)
'''
########train = train[top_features_w_TripType]
unique_visits = test['VisitNumber'] # to be used later in output file
########test = test[top_features]

########y_train = train['TripType'].values
# top features doesn't have VisitNumber in it, so no need to drop
########X_train = train.drop(['TripType'], axis = 1).values
########X_test = test.values
#%%
#clf = linear_model.logistic.LogisticRegression(C = 1, n_jobs = -1, 
#                                               class_weight = 'balanced')
scores = cross_validation.cross_val_score(clf, X_train, y_train, cv = 5)
print 'cross_val score', scores, scores.mean()
#%%
clf.fit(X_train, y_train)
predicted_train_classes = clf.predict(X_train)
print 'training error', metrics.accuracy_score(y_train, predicted_train_classes)
#%%
# Training error = 0.91. Cross-valid error = 0.64 We are likely
# over-fitting (ie. high variance).  Reducing max. depth
pars = {'max_depth': [5, 10, 15, 20]}
clf = grid_search.GridSearchCV(clf, param_grid = pars, verbose = 5) #RandomizedSearchCV(clf, pars)
clf.fit(X_train, y_train)
#%%
predicted_train_classes = clf.predict(X_train)
print 'training error (after tuning)', metrics.accuracy_score(y_train, predicted_train_classes)


#%%
predicted_proba = clf.predict_proba(X_test)
print predicted_proba.shape
#%%

unique_trip_types =  np.sort(train['TripType'].unique())
unique_trip_types =  ['TripType_'+ str(int(t)) for t in unique_trip_types]
print unique_trip_types

df_predicted = pd.DataFrame(predicted_proba, columns = unique_trip_types)
df_unique_visits = pd.DataFrame(unique_visits).reset_index(drop = True)
df_predicted = pd.concat([df_unique_visits, df_predicted], axis = 1)
print df_predicted.shape
df_predicted.to_csv('./data/test_predicted.csv', index = False)

#%% error analysis
# confusion matrix
cm = metrics.confusion_matrix(y_train, predicted_train_classes)
plt.figure(figsize = (10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
unique_trip_types =  np.sort(train['TripType'].unique())
unique_trip_types = unique_trip_types.astype(int)
tick_marks = np.arange(len(unique_trip_types))
plt.xticks(tick_marks, unique_trip_types, rotation=0)
plt.yticks(tick_marks, unique_trip_types)
plt.colorbar() 