import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Load dataset
df = pd.read_csv("G:\My Drive\Docs\Sid\Career\MSITM\Software Design for Data Scientists\Project\Project files\heart-risk-predictor\Data\heart_data.csv")
# df = df[['age', 'cholesterol', 'weight', 'gluc', 'ap_lo', 'ap_hi', 'active', 'height', 'gender', 'smoke', 'alco', 'cardio']]
df = df.drop(columns=['index', 'id'])  # Drop unused columns

# Split features and target
X = df.drop('cardio', axis=1)
y = df['cardio']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict class labels for accuracy
y_pred_class = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy Score: {accuracy:.4f}")
