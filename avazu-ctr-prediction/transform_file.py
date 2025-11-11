# ==========================================================
# IMPORT LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# LOAD DATA (LIMIT TO 100K ROWS FOR SPEED)
# ==========================================================
train = pd.read_csv(
    '/Users/john/PycharmProjects/click-through-rate-prediction/avazu-ctr-prediction/train.gz',
    compression='gzip',
    nrows=100000
)
test = pd.read_csv(
    '/Users/john/PycharmProjects/click-through-rate-prediction/avazu-ctr-prediction/test.gz',
    compression='gzip',
    nrows=100000
)

print("Train columns:", train.columns.tolist())
print("Train shape:", train.shape)
print(train.head())

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
df = train.copy()

# Convert hour to datetime
df['hour'] = pd.to_datetime(df['hour'], format='%y%m%d%H', errors='coerce')

# Extract time-based features
df['dayofweek'] = df['hour'].dt.dayofweek
df['hourofday'] = df['hour'].dt.hour

# Drop high-cardinality or identifier columns
df.drop(columns=['hour', 'id', 'device_ip', 'device_id'], inplace=True, errors='ignore')

# ==========================================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# ==========================================================
# TRAIN-TEST SPLIT
# ==========================================================
X = df.drop('click', axis=1)
y = df['click']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================================
# MODEL TRAINING WITH XGBOOST
# ==========================================================
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=42,
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ==========================================================
# EVALUATION
# ==========================================================
y_pred = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
print(f"AUC Score: {auc:.4f}")

# ==========================================================
# CLEAN NUMERIC STRINGS (SAFETY STEP FOR SHAP)
# ==========================================================
def clean_numeric_strings(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[\[\]]', '', regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

X_train = clean_numeric_strings(X_train)
X_test = clean_numeric_strings(X_test)

# ==========================================================
# SHAP EXPLAINABILITY
# ==========================================================
# Sample subset for faster computation
X_sample = X_test.sample(200, random_state=42)


# ✅ Use a prediction function explicitly instead of the model object
explainer = shap.Explainer(model.predict_proba, X_sample)

# Compute SHAP values (for the "clicked" class = 1)
shap_values = explainer(X_sample)

# Summary plots
shap.summary_plot(shap_values[..., 1], X_sample)
shap.summary_plot(shap_values[..., 1], X_sample, plot_type="bar")