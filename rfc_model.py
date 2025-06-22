import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and prepare data
df = pd.read_csv("Data\heart_data.csv")
df['age_years'] = (df['age'] / 365).astype(int)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df = df.drop(columns=['index', 'id', 'age', 'weight', 'height'])

X = df.drop('cardio', axis=1)
y = df['cardio']

# Sample for faster testing
df_sample = df.sample(n=10000, random_state=42)
X_sample = df_sample.drop('cardio', axis=1)
y_sample = df_sample['cardio']

X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
