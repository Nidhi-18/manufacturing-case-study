# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:01:56 2019

@author: Nidhi123
"""

import pandas as pd
import numpy as np

df=pd.read_csv("case_3_LR_manufacturing.csv")
df.columns=['Cement','Blast Furnace slag','Fly Ash','Water','Superplasticizers','CoarseAgg','FineAgg','Age(day)','Concrete_strength']
#types datatypes 
df.dtypes
#statistical datas
df.describe()
#checking number of zeros
(df['Cement'] == 0).astype(int).sum(axis=0)# 0 zeros
(df['Blast Furnace slag'] == 0).astype(int).sum(axis=0)# 471 zeros
(df['Fly Ash'] == 0).astype(int).sum(axis=0)# 566 zeros
(df['Water'] == 0).astype(int).sum(axis=0)# 0 zeros
(df['Superplasticizers'] == 0).astype(int).sum(axis=0)# 379 zeros
(df['CoarseAgg'] == 0).astype(int).sum(axis=0)# 0 zeros
(df['FineAgg'] == 0).astype(int).sum(axis=0)# 0 zeros
(df['Age(day)'] == 0).astype(int).sum(axis=0)# 0 zeros
(df['Concrete_strength'] == 0).astype(int).sum(axis=0)# 0 zeros
#describe any one column
df['Superplasticizers'].describe()
df['Fly Ash'].describe()
df['Blast Furnace slag'].describe()

# list of outliers 
def outliers_iqr(df):
    quartile_1, quartile_3 = np.percentile(df, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((df > upper_bound) | (df < lower_bound))


outliers_iqr(df['Cement'])
outliers_iqr(df['Blast Furnace slag'])
outliers_iqr(df['Fly Ash'])
outliers_iqr(df['Water'])
outliers_iqr(df['Superplasticizers'])
outliers_iqr(df['CoarseAgg'])
outliers_iqr(df['FineAgg'])
outliers_iqr(df['Age(day)'])
outliers_iqr(df['Concrete_strength'])

#removing outliers
q1=df.quantile(0.25)
q3=df.quantile(0.75)
iqr=q3-q1
df_0=df[~((df<(q1-1.5*iqr))|(df>(q3+1.5*iqr))).any(axis=1)]
df_final=df_0.reset_index(drop=True)


df_final_code=df_final.to_csv('_clean_data_case3.csv_')



################LINEAR REGRESSION MODEL(ML)##################################
from sklearn.linear_model import LinearRegression# import model
from sklearn.model_selection import train_test_split
#from sklearn.datasets import load_iris
train_set, test_set=train_test_split(df_final, test_size=0.2, random_state=70)

######## separating dataset for train #########
X=train_set.drop('Concrete_strength', axis=1)
y=train_set['Concrete_strength']


########## separating dataset for test ##########
X_test=test_set.drop('Concrete_strength',axis=1)
y_test=test_set['Concrete_strength']



'''################### PREDICTING VALUES FOR TRAINING DATASET ###########################'''

''' ________________FOR LINEAR REGRESSION MODEL___________________'''

lin_reg = LinearRegression()# instantiate
lin_reg.fit(X, y)# fit the model to the training data (learn the coefficients)

from sklearn.metrics import mean_squared_error
y_pred=lin_reg.predict(X)
lin_mse=mean_squared_error(y, y_pred)
lin_rmse=np.sqrt(lin_mse)
lin_rmse
'''---------------- R-mean Square Value ---------------'''
lin_rq=lin_reg.score(X, y)


'''______________________FOR RIDGE REGRESSION MODEL__________________'''

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X,y)

y_ridge = ridge_reg.predict(X)
lin_mse_ridge = mean_squared_error(y, y_ridge)
lin_rmse_ridge = np.sqrt(lin_mse_ridge)
lin_rmse_ridge
''' ----------------- R- mean Square Value -----------------------'''
ridge_rq=ridge_reg.score(X,y)



'''________________FOR LASSO REGRESSION MODEL______________________'''

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)

y_lasso = lasso_reg.predict(X)
lin_mse_lasso = mean_squared_error(y, y_lasso)
lin_rmse_lasso = np.sqrt(lin_mse_lasso)
lin_rmse_lasso
'''------------------R- Square Value --------------------------'''
lasso_rq=lasso_reg.score(X, y)


'''__________________FOR ELASTO NET MODEL________________________'''

from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)

y_elastic_net = elastic_net.predict(X)
lin_mse_elastic_net = mean_squared_error(y, y_elastic_net)
lin_rmse_elastic_net = np.sqrt(lin_mse_elastic_net)
lin_rmse_elastic_net
'''-------------------R- Squared Value ---------------------'''
elastic_net_rq=elastic_net.score(X, y)





'''__________________FOR SGD REGRESSOR________________________'''

from sklearn import linear_model
clf = linear_model.SGDRegressor(max_iter=10000, tol=1e-3)
clf.fit(X,y)

from sklearn.metrics import mean_squared_error
y_pred=clf.predict(X)
clf_mse=mean_squared_error(y, y_pred)
clf_rmse=np.sqrt(clf_mse)
clf_rmse
'''-------------------R-square Value--------------------------'''
clf_rq=clf.score(X, y)




'''################### PREDICTING VALUES FOR TESTING DATASET ###################'''

'''___________________MODEL FOR RIDGE REGRESSION __________________'''
y_ridge_test = ridge_reg.predict(X_test)
lin_mse_ridge_test = mean_squared_error(y_test, y_ridge_test)
lin_rmse_ridge_test = np.sqrt(lin_mse_ridge_test)
lin_rmse_ridge_test
'''------------R-Squared Value --------------'''
ridge_test_rq=ridge_reg.score(X_test, y_test)


'''___________MODEL FOR LINEAR REGRESSION______________'''

y_lin_test = lin_reg.predict(X_test)
lin_mse_lin_test = mean_squared_error(y_test, y_lin_test)
lin_rmse_lin_test = np.sqrt(lin_mse_lin_test)
lin_rmse_lin_test
'''-------------------R- Square Value -----------------'''
lin_test_rq=lin_reg.score(X_test, y_test)


'''_______________MODEL FOR LASSO REGRESSION _________________'''

y_lasso_test = lasso_reg.predict(X_test)
lasso_mse_lasso_test = mean_squared_error(y_test, y_lasso_test)
lasso_rmse_lasso_test = np.sqrt(lasso_mse_lasso_test)
lasso_rmse_lasso_test
'''--------------------R-Square Value ------------------------'''
lasso_test_rq=lasso_reg.score(X_test, y_test)



'''______________FOR ELASTO NET REGRESSION ____________'''

y_elastic_net_test = elastic_net.predict(X_test)
elastic_net_mse_elastic_net_test = mean_squared_error(y_test, y_elastic_net_test)
elastic_net_rmse_elastic_net_test = np.sqrt(elastic_net_mse_elastic_net_test)
elastic_net_rmse_elastic_net_test
'''-------------------R- Square Value ----------------------'''
elastic_net_test_rq=elastic_net.score(X_test, y_test)


'''_________________FOR SGD REGRESSION__________________'''
y_clf_test=clf.predict(X_test)
clf_mse_test=mean_squared_error(y_test, y_clf_test)
clf_rmse_clf_test=np.sqrt(clf_mse_test)
clf_rmse_clf_test

'''-----------------R-square-------------------'''
clf_rq_test=clf.score(X_test, y_test)

##############creating table dataframe############################

dict_rmse={'rmse_train_columns': [lin_reg, ridge_reg, lasso_reg, elastic_net, clf],  
           'rmse_train_values':[lin_rmse, lin_rmse_ridge, lin_rmse_lasso, lin_rmse_elastic_net, clf_rmse],
           'R-square_train': [lin_rq, ridge_rq, lasso_rq, elastic_net_rq, clf_rq]}
df1_train=pd.DataFrame.from_dict(dict_rmse)


dict_r_sq={'rmse_test_columns': [lin_reg, ridge_reg, lasso_reg, elastic_net, clf],
           'rmse_test_values': [lin_rmse_lin_test, lin_rmse_ridge_test, lasso_rmse_lasso_test, elastic_net_rmse_elastic_net_test, clf_rmse_clf_test],
           'R-square_test': [lin_test_rq, ridge_test_rq, lasso_test_rq, elastic_net_test_rq, clf_rq_test]}
df1_test=pd.DataFrame.from_dict(dict_r_sq)


final_df=df1_train.join(df1_test, lsuffix='t', rsuffix='t')
train_test_sheet=final_df.to_csv("train_test_modify_caseL3.csv")




