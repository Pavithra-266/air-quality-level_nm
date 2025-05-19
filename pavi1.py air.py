# air_quality_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('air_quality_data.csv')  # Replace with your actual dataset

# Explore and clean
print(df.head())
print(df.info())

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Example features and target
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
target = 'AQI_Category'  # Assuming AQI is categorized as Good, Moderate, etc.

# Encode categorical target if needed
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(df[features])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
