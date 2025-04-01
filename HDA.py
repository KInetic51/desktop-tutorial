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
    print("First 5 rows:\n", data.head())
    print("Basic statistics:\n", data.describe())
    print("Class distribution:\n", data['Outcome'].value_counts())
    return data

def impute_zeros_by_class(df, feature):
    median_malignant = df[(df['Outcome'] == 1) & (df[feature] != 0)][feature].median()
    median_benign = df[(df['Outcome'] == 0) & (df[feature] != 0)][feature].median()
    df.loc[(df['Outcome'] == 1) & (df[feature] == 0), feature] = median_malignant
    df.loc[(df['Outcome'] == 0) & (df[feature] == 0), feature] = median_benign
    return df

def preprocess_data(data):
    features_to_impute = ['CellSizeMean', 'Perimeter', 'Texture']  # Update based on dataset features
    for feature in features_to_impute:
        data = impute_zeros_by_class(data, feature)
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns

def split_data(X, y, test_size=0.25, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

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
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Test set accuracy of optimized model:", best_model.score(X_test, y_test))
    return best_model

def prune_decision_tree(best_model, X_train, y_train, X_test, y_test):
    path = best_model.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[:-1]
    scores = [DecisionTreeClassifier(random_state=42, ccp_alpha=alpha).fit(X_train, y_train).score(X_test, y_test) for alpha in alphas]
    best_alpha = alphas[scores.index(max(scores))]
    final_model = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    final_model.fit(X_train, y_train)
    print("Optimal pruning alpha:", best_alpha)
    print("Final model accuracy after pruning:", final_model.score(X_test, y_test))
    return final_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    print("ROC AUC:", roc_auc)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()

def extract_decision_rules(model, feature_names):
    from sklearn.tree import _tree
    tree_ = model.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined" for i in tree_.feature]
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
    print("Extracted Decision Rules:")
    for rule in paths:
        print(rule)
    return paths

def save_model(model, filename='cancer_decision_tree_model.pkl'):
    joblib.dump(model, filename)
    print("Model saved as", filename)

def main():
    filepath = 'cancer issue.csv'
    data = load_and_explore_data(filepath)
    X, y, feature_names = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    best_model = optimize_decision_tree(X_train, y_train, X_test, y_test)
    final_model = prune_decision_tree(best_model, X_train, y_train, X_test, y_test)
    evaluate_model(final_model, X_test, y_test)
    extract_decision_rules(final_model, feature_names)
    save_model(final_model)

if __name__ == '__main__':
    main()
