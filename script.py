import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean():.3%}")

# Feature engineering
df['hour'] = (df['Time'] // 3600) % 24
df['high_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

# Select top fraud features (from literature)
features = ['Amount', 'V17', 'V12', 'V14', 'V10', 'V16', 'V11', 'hour']
X = df[features].fillna(0)

# Isolation Forest (unsupervised anomaly detection)
model = IsolationForest(contamination=0.0008, random_state=42, n_jobs=-1)
df['anomaly_score'] = model.fit_predict(X)
df['fraud_predicted'] = (df['anomaly_score'] == -1).astype(int)

# Results
precision = precision_score(df['Class'], df['fraud_predicted'])
print(f"\n=== RESULTS ===")
print(f"Precision: {precision:.1%}")
print(f"Detected fraud: {df['fraud_predicted'].sum()}")
print("\nClassification Report:")
print(classification_report(df['Class'], df['fraud_predicted']))

# Save results
df[['Time', 'Amount', 'Class', 'fraud_predicted', 'anomaly_score', 'hour']].to_csv('fraud_analysis.csv', index=False)
print("\nSaved: fraud_analysis.csv")

# Quick visualization
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
sns.histplot(df[df['Class']==1]['Amount'], bins=50, alpha=0.7, label='Fraud')
sns.histplot(df[df['Class']==0]['Amount'], bins=50, alpha=0.7, label='Normal')
plt.legend(); plt.title('Amount Distribution')

plt.subplot(1,2,2)
fraud_by_hour = df.groupby('hour')['Class'].mean()
plt.plot(fraud_by_hour.index, fraud_by_hour.values)
plt.title('Fraud Rate by Hour'); plt.xlabel('Hour'); plt.ylabel('Fraud Rate')
plt.savefig('fraud_plots.png')
plt.show()