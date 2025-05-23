import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams["figure.figsize"] = (20,10)

# Load the dataset
df1 = pd.read_csv(r"G:\python_pycharm\machine learning\project\BangloreHomePrices\model\Bengaluru_House_Data.csv")

# Drop features that are not required to build our model
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')

# Data Cleaning: Handle NA values
df3 = df2.dropna()

# Feature Engineering
# Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
df3.loc[:, 'bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))


# Explore total_sqft feature
def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

print(df3[~df3['total_sqft'].apply(lambda x: is_float(x))].head(10))



# Convert sqft ranges to their average
def convert_sqft_to_num(x):
    try:
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
df4 = df4[df4['total_sqft'].notnull()]

# Add new feature called price per square feet
df5 = df4.copy()
df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']

df5_stats = df5['price_per_sqft'].describe()

df5.to_csv("bhp.csv", index=False)

# Clean location feature
df5['location'] = df5['location'].apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)

# Dimensionality Reduction
location_stats_less_than_10 = location_stats[location_stats <= 10]

df5['location'] = df5['location'].apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

# Outlier Removal Using Business Logic
df6 = df5[~((df5['total_sqft'] / df5['bhk']) < 300)]


# Outlier Removal Using Standard Deviation and Mean
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf['price_per_sqft'])
        st = np.std(subdf['price_per_sqft'])
        reduced_df = subdf[(subdf['price_per_sqft'] > (m - st)) & (subdf['price_per_sqft'] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7 = remove_pps_outliers(df6)

# Remove bhk outliers
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df['price_per_sqft']),
                'std': np.std(bhk_df['price_per_sqft']),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df['price_per_sqft'] < stats['mean']].index.values)
    return df.drop(exclude_indices, axis='index')

df8 = remove_bhk_outliers(df7)

# Visualizations
plt.hist(df8['price_per_sqft'], rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
plt.show()

plt.hist(df8['bath'], rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
plt.show()

# Remove outliers based on number of bathrooms
df9 = df8[df8['bath'] < df8['bhk'] + 2]
df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')

# One Hot Encoding For Location
dummies = pd.get_dummies(df10['location'])
df11 = pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df12 = df11.drop('location', axis='columns')

# Build a Model Now...
X = df12.drop(['price'], axis='columns')
y = df12['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
lr_clf.score(X_test, y_test)

# Use K Fold cross validation
from sklearn.model_selection import ShuffleSplit, cross_val_score
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv)

# Find best model using GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'splitter': ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })
    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

find_best_model_using_gridsearchcv(X, y)

import os
import pickle
import json

# Create directory if it doesn't exist
artifacts_dir = os.path.join('server', 'artifacts')
os.makedirs(artifacts_dir, exist_ok=True)

# Save the trained model
model_path = os.path.join(artifacts_dir, 'bangalore_home_prices_model.pickle')
with open(model_path, 'wb') as f:
    pickle.dump(lr_clf, f)
print(f"Model saved successfully at {model_path}")

# Save the columns
columns = {
    'data_columns': [col.lower() for col in X.columns]
}
columns_path = os.path.join(artifacts_dir, 'columns.json')
with open(columns_path, 'w') as f:
    json.dump(columns, f)
print(f"Columns saved successfully at {columns_path}")

