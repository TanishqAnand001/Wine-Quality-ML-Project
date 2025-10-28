import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

try:
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(data_url, sep=';')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
df = df.drop('quality', axis=1)

X = df.drop('quality_label', axis=1)
y = df['quality_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

preds = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, preds)}")

joblib.dump(model, 'api/wine_model.joblib')
joblib.dump(scaler, 'api/wine_scaler.joblib')

print("Model and scaler saved as wine_model.joblib and wine_scaler.joblib")