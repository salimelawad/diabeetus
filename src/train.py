import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import logging
import pickle

logging.basicConfig(level=logging.DEBUG)

diabetes = pd.read_csv('data/diabetes.csv')

logging.warn(diabetes.head())


X = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']]

mean =  X.mean().to_dict()
min_ = X.min().to_dict()
max_ = X.max().to_dict()

logging.debug(mean)

y = diabetes[['Outcome']]

logging.debug(X.head())
logging.debug(y.head())

fit = LinearRegression(fit_intercept=True, normalize=True, copy_X=True).fit(X, y)
# fit = RandomForestClassifier().fit(X, y)

y_pred = fit.predict(X)

logging.debug(np.transpose(np.rint(y_pred)))
logging.debug(fit.score(X, y))

pickle.dump(fit, open('model/model.p','wb'))
pickle.dump(mean, open('model/mean.p','wb'))
pickle.dump(min_, open('model/min.p','wb'))
pickle.dump(max_, open('model/max.p','wb'))