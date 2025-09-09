import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def train_logistic_regression(df, label_col='label', positive_label='KO'):
    """
    Trains a logistic regression model to classify OK vs KO.

    Args:
        df (pd.DataFrame): DataFrame with features + label column
        label_col (str): Name of label column (default 'label')
        positive_label (str): The value representing 'KO' (default 'KO')

    Returns:
        model (LogisticRegression): The trained model
        report (str): Scikit-learn classification report
        accuracy (float): Accuracy on test set
        feature_importance (pd.DataFrame): Feature coefficients
    """
    # Separate features and labels
    X = df.drop(columns=[label_col])
    y = (df[label_col] == positive_label).astype(int)  # 1 for KO, 0 for OK

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classification report & accuracy
    report = classification_report(y_test, y_pred, target_names=['OK', 'KO'])
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importance (coefficients)
    importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values(by='coefficient', key=abs, ascending=False)

    return model, report, accuracy, importance

# Optional: For standalone test/demo
if __name__ == "__main__":
    # Change to your actual CSV path and label column as needed!
    df = pd.read_csv("dataset/combined_labeled.csv")
    model, report, acc, importance = train_logistic_regression(df)
    print("Classification report:\n", report)
    print("Accuracy:", acc)
    print("Most discriminative features:\n", importance)
