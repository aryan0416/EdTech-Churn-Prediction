import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('edtech_users.csv')

le = LabelEncoder()
df['SubscriptionTier'] = le.fit_transform(df['SubscriptionTier'])

X = df[['DaysSinceLastLogin', 'VideoCompletionRate', 'SubscriptionTier']]
y = df['ChurnLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

feature_importances = model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

plt.figure(figsize=(8, 5))
plt.bar(X.columns, feature_importances, color='skyblue')
plt.title('Feature Importances for Churn Prediction')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
