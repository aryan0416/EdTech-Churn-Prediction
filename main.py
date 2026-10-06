from src.preprocess import load_and_preprocess
from src.model import train_and_evaluate, plot_feature_importance

def main():
    df = load_and_preprocess('data/edtech_users.csv')
    features, importances = train_and_evaluate(df)
    plot_feature_importance(features, importances, 'feature_importance.png')

if __name__ == "__main__":
    main()
