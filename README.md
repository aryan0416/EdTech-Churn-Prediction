# EdTech Churn Prediction

![EdTech Banner](banner.jpg)

## Business Problem
In the EdTech industry, the cost of acquiring a new customer is significantly higher than retaining an existing one. High churn rates directly erode revenue and indicate potential issues with user engagement or content quality. This project focuses on predicting which users are most likely to churn based on their platform activity and subscription details. By identifying at-risk users, the business can proactively offer targeted retention discounts and personalized interventions.

## Tech Stack
* **Python**: Core programming language.
* **Pandas**: For dataset manipulation.
* **Scikit-Learn**: For data splitting, label encoding, and training the Random Forest Classifier.
* **Matplotlib**: For visualizing feature importance.

## Methodology
1. **Data Ingestion**: The script loads user engagement and subscription data.
2. **Data Preprocessing**: Categorical features like Subscription Tier are converted into numerical formats using Label Encoding.
3. **Model Training**: A Random Forest Classifier is trained on historical data, learning the complex non-linear relationships between engagement metrics and the likelihood of churning.
4. **Evaluation**: The model is evaluated on a hold-out test set to determine its accuracy.
5. **Feature Importance**: The script extracts and visualizes which features (e.g., Days Since Last Login vs. Video Completion Rate) are the strongest predictors of churn, saving the plot as `feature_importance.png`.

## Business Impact
A robust predictive model for churn allows for targeted retention strategies:
* Maximized Customer Lifetime Value (CLTV) by reducing turnover.
* Optimized marketing budgets by directing retention discounts only to high-risk users rather than a blanket approach.
* Actionable product insights derived from feature importance (e.g., discovering that low video completion is the primary driver of churn).

## Setup Instructions
**Prerequisites:**
You must provide the dataset `edtech_users.csv` in the root of this directory. The CSV must contain the following columns: `DaysSinceLastLogin`, `VideoCompletionRate`, `SubscriptionTier`, and `ChurnLabel`.

**Execution:**
Run the analysis script using the following command:
```bash
python main.py
```
This will print the model's accuracy, output the importance of each feature, and generate a visualization of these importances.
