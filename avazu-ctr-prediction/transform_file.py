import pandas as pd

train = pd.read_csv('/Users/john/PycharmProjects/click-through-rate-prediction/avazu-ctr-prediction/train.gz', compression='gzip', nrows=100000)
test = pd.read_csv('/Users/john/PycharmProjects/click-through-rate-prediction/avazu-ctr-prediction/test.gz', compression='gzip', nrows=100000)

print("Train columns:", train.columns.tolist())
print("Train shape:", train.shape)
print(train.head())

df = train.copy()

df['hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H')
df['dayofweek'] = df['hour'].dt.dayofweek
df['hourofday'] = df['hour'].dt.hour
print(df[['device_ip', 'device_id']])
df.drop(columns=['hour', 'id', 'device_ip', 'device_id'], inplace=True)

print(df)

from sklearn.preprocessing import LabelEncoder

for col in df.columns:
    if df[col].dtype == 'object':
        print(col)
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

from sklearn.model_selection import train_test_split

X = df.drop('click', axis=1)
print(X)
y = df['click']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb
from sklearn.metrics import roc_auc_score

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc:.4f}")