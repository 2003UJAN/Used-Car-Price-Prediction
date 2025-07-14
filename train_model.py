import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

df = pd.read_csv("car data.csv")

# Convert columns to numerical
def extract_number(value, unit):
    try:
        return float(value.replace(unit, '').strip())
    except:
        return np.nan

df['mileage'] = df['mileage'].str.extract(r'([\d.]+)').astype(float)
df['engine'] = df['engine'].str.extract(r'([\d.]+)').astype(float)
df['max_power'] = df['max_power'].str.extract(r'([\d.]+)').astype(float)

# Handle missing values
df['mileage'].fillna(df['mileage'].median(), inplace=True)
df['engine'].fillna(df['engine'].median(), inplace=True)
df['max_power'].fillna(df['max_power'].median(), inplace=True)
df['seats'].fillna(df['seats'].mode()[0], inplace=True)

# Drop unusable columns
df.drop(['torque', 'name'], axis=1, errors='ignore', inplace=True)

# Car age
df['Car_Age'] = 2025 - df['year']
df.drop(['year'], axis=1, inplace=True)

# Encode categoricals
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop("selling_price", axis=1)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=None)
model.fit(X_train, y_train)
print("RMSE:", mean_squared_error(y_test, model.predict(X_test), squared=False))

# Save model and columns
joblib.dump(model, 'car_price_predictor.pkl')
joblib.dump(X.columns.tolist(), 'model_features.pkl')

