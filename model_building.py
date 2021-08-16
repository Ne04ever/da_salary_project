# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('eda_data.csv')
df.columns


df_model = df[['Rating','Type of ownership', 'Industry', 'Sector', 'Revenue','min_salary', 'max_salary', 'avg_clean','state', 'company_age', 'python', 'sql', 'r', 'aws','num_compitators']]

#get dummy 

df_dum = pd.get_dummies(df_model)

#train test split
from sklearn.model_selection import train_test_split
X = df_dum.drop('avg_clean',axis=1)
y = df_dum['avg_clean']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#build models
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression , Lasso
ln = LinearRegression()
ln.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(ln,X_train, y_train, scoring='neg_mean_absolute_error', cv = 3))

#lasso
lm_1 = Lasso(0.2)
lm_1.fit(X_train,y_train)
np.mean(cross_val_score(lm_1,X_train, y_train, scoring='neg_mean_absolute_error', cv = 3))

alpha = []
error = []

for i in range(1,10):
    alpha.append(i/10)
    ln1 = Lasso(alpha=(i/10))
    error.append(np.mean(cross_val_score(ln1,X_train, y_train, scoring='neg_mean_absolute_error', cv = 3)))

plt.plot(alpha,error)    
err = tuple(zip(alpha,error))
err_df = pd.DataFrame(err) 


#random forrest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
np.mean(cross_val_score(rf,X_train, y_train, scoring='neg_mean_absolute_error', cv = 3))

#tuning using GridSearchCv
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mae','mse'), 'max_features':('sqrt','log2','auto')}

gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error', cv = 3)
gs.fit(X_train,y_train)
gs.best_score_
gs.best_estimator_


#prediction
tpred_ln = ln.predict(X_test)
tpred_lm_1 = lm_1.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_ln)
mean_absolute_error(y_test,tpred_lm_1)
mean_absolute_error(y_test,tpred_rf)
