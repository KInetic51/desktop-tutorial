import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')


def load_and_explore_data(filepath):
    data = pd.read_csv(filepath)
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nBasic statistics:")
    print(data.describe())
    print("\nClass distribution:")
    print(data['Outcome'].value_counts())
    return data


def impute_zeros_by_class(df, feature):
    # Calculate the median for non-zero values for each class (Outcome)
    diabetic_median = df[(df['Outcome'] == 1) & (df[feature] != 0)][feature].median()
    non_diabetic_median = df[(df['Outcome'] == 0) & (df[feature] != 0)][feature].median()
    df.loc[(df['Outcome'] == 1) & (df[feature] == 0), feature] = diabetic_median
    df.loc[(df['Outcome'] == 0) & (df[feature] == 0), feature] = non_diabetic_median
    return df


def preprocess_data(data):
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in zero_features:
        data = impute_zeros_by_class(data, feature)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns


def split_data(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def optimize_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    print("\nBest parameters found:")
    print(grid_search.best_params_)
    test_accuracy = best_dt.score(X_test, y_test)
    print(f"Accuracy of optimized decision tree: {test_accuracy:.4f}")
    return best_dt


def prune_decision_tree(best_dt, X_train, y_train, X_test, y_test):
    path = best_dt.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[:-1]  # Exclude the maximum alpha which yields a trivial tree
    test_scores = []
    for alpha in alphas:
        dt = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        dt.fit(X_train, y_train)
        test_scores.append(dt.score(X_test, y_test))
    best_alpha = alphas[test_scores.index(max(test_scores))]
    final_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    final_dt.fit(X_train, y_train)
    final_accuracy = final_dt.score(X_test, y_test)
    print(f"\nOptimal alpha for pruning: {best_alpha:.6f}")
    print(f"Accuracy of final pruned decision tree: {final_accuracy:.4f}")
    return final_dt, best_alpha


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    print("\nEvaluation Metrics:")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("ROC AUC:", roc_auc)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def extract_decision_rules(model, feature_names):
    from sklearn.tree import _tree
    tree_ = model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature
    ]
    paths = []

    def recurse(node, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_path = path.copy()
            left_path.append(f"({name} <= {threshold:.2f})")
            recurse(tree_.children_left[node], left_path)
            right_path = path.copy()
            right_path.append(f"({name} > {threshold:.2f})")
            recurse(tree_.children_right[node], right_path)
        else:
            path_str = "IF " + " AND ".join(path) + f" THEN predict: {np.argmax(tree_.value[node])}"
            paths.append(path_str)

    recurse(0, [])
    print("\nExtracted Decision Rules:")
    for rule in paths:
        print(rule)
    return paths


def save_model(model, filename='diabetes_decision_tree_model.pkl'):
    joblib.dump(model, filename)
    print("\nFinal model saved as", filename)


def main():
    # Load and explore data
    data = load_and_explore_data('diabetes.csv')

    # Preprocess data
    X, y, feature_names = preprocess_data(data)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Optimize decision tree using grid search
    best_dt = optimize_decision_tree(X_train, y_train, X_test, y_test)

    # Prune the decision tree
    final_dt, best_alpha = prune_decision_tree(best_dt, X_train, y_train, X_test, y_test)

    # Evaluate the final model
    evaluate_model(final_dt, X_test, y_test)

    # Extract and display decision rules
    extract_decision_rules(final_dt, feature_names)

    # Save the final model to disk
    save_model(final_dt)


if __name__ == '__main__':
    main()
