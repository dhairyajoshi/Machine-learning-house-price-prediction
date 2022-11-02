import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import pandas as pd

def calc_mae(n):
    model = RandomForestRegressor(n_estimators=n,random_state=23)
    model.fit(X_train, y_train)

    prediction = model.predict(X_valid)

    return mae(y_valid, prediction)


data = pd.read_csv('./melb_data.csv')

y= data['Price']
X = data.drop(['Price','Lattitude','Longtitude'], axis=1)
num_features = [col for col in X if X[col].dtype in ['int', 'float64']]
X= X[num_features]

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8)

imputer = SimpleImputer(strategy='median')
X_train= pd.DataFrame(imputer.fit_transform(X_train))
X_valid= pd.DataFrame(imputer.transform(X_valid))

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_valid)


best = min([calc_mae(i) for i in range(100,600,50)])

print('best minimum absolute difference: ', best)
# best score = 185005.172