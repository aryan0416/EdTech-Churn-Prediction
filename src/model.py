from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_and_evaluate(df):
    X = df[['Last Interaction', 'Usage Frequency', 'Subscription Type']]
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    
    feature_importances = model.feature_importances_
    for feature, importance in zip(X.columns, feature_importances):
        print(f"{feature}: {importance:.4f}")
        
    return X.columns, feature_importances

def plot_feature_importance(features, importances, output_path):
    plt.figure(figsize=(8, 5))
    plt.bar(features, importances, color='skyblue')
    plt.title('Feature Importances for Churn Prediction')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
