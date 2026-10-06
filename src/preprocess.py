import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    le = LabelEncoder()
    df['Subscription Type'] = le.fit_transform(df['Subscription Type'])
    df = df.dropna(subset=['Last Interaction', 'Usage Frequency', 'Subscription Type', 'Churn'])
    return df
