import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load The data

data_df=pd.read_csv("train-data.csv")

#%% Trimmed un wanted features

data_df.drop('New_Price', axis=1, inplace=True)
data_df.drop('Unnamed', axis=1, inplace=True)

#%% Cleaning  Training data

data_df['Mileage']=data_df['Mileage'].str.replace('kmpl','')
data_df['Mileage']=data_df['Mileage'].str.replace('km/kg','')
data_df['Engine']=data_df['Engine'].str.replace('CC','')
data_df['Power']=data_df['Power'].str.replace('bhp','')
data_df['Transmission']=data_df['Transmission'].str.replace('Manual','0')
data_df['Transmission']=data_df['Transmission'].str.replace('Automatic','1')

#%% Converting Strings (Objects) into
arr=['Mileage','Transmission','Engine','Power']
for i in arr:
    data_df[i]=data_df[i].astype(float)

#%%
print(data_df.info())
#%%
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

encoded_cols = ohe.fit_transform(data_df[['Fuel_Type', 'Owner_Type', 'Name', 'Location']]).toarray()
encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(['Fuel_Type', 'Owner_Type', 'Name', 'Location']))
data_df = data_df.drop(['Fuel_Type', 'Owner_Type', 'Name', 'Location'], axis=1)
data_df = pd.concat([data_df, encoded_df], axis=1)


#%% Removing zeros from the data
data_df = data_df.fillna(0)
arr=['Mileage','Engine','Seats']
for i in arr:
    mean_value = data_df.loc[data_df[i] != 0, i].mean()
    data_df[i] = data_df[i].replace('nan', mean_value)
#%% Visualize the dataset
data_df.info()
#data_df['Year']=data_df['Year']%2000
#%%
# Select the input features (X) and target variable (y)
X = data_df.drop('Price', axis=1)
y = data_df['Price']
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%%Train the model 

import xgboost as xgb
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
#%%Test
y_pred = model.predict(X_test)
#%%
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
r2 = r2_score(y_test, y_pred)
print("R2 score: %.2f" % r2)
