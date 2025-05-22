import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv('Cancer_Data.csv')

# Drop the ID column and unnamed last column
data = data.drop(columns=[data.columns[0], data.columns[-1]])

# Display summary statistics
print(data.describe())

# Check for duplicate rows
print("Duplicated rows:", data.duplicated(keep=False).sum())

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Separate features and target variable
X = data.iloc[:, 1:]
y = data.iloc[:, 0].to_numpy()

# Display dataset info
print(X.info())

# Plot feature distributions
fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(15, 30))
for i, column in enumerate(X.columns):
    ax = axes[i // 3, i % 3]
    X[column].plot(kind='hist', ax=ax, title=column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
plt.tight_layout()
plt.show()

# Display class distribution
class_counts = data['diagnosis'].value_counts()
plt.figure(figsize=(8, 6))
class_counts.plot(kind='bar', color=['orange', 'royalblue'])
plt.text(0, class_counts[0], str(class_counts[0]), ha='center', va='bottom')
plt.text(1, class_counts[1], str(class_counts[1]), ha='center', va='bottom')
plt.xlabel('Class')
plt.ylabel('Quantity')
plt.title('Class Distribution')
plt.show()

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

# Apply PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['target'] = y
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='target')
plt.title('PCA Visualization')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Dummy classifier baseline
dummy = DummyClassifier(strategy='most_frequent', random_state=60)
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, y_pred))

# Decision Tree Classifier with Grid Search
print("\n--- Decision Tree Classifier ---")
dt_clf = DecisionTreeClassifier(random_state=60)
dt_param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}
grid_search = GridSearchCV(dt_clf, dt_param_grid, cv=5)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))

# Confusion matrix function
def plot_confusion_matrix(y_true, y_pred, title, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion matrix for {title}')
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.show()

plot_confusion_matrix(y_test, y_pred, 'DecisionTreeClassifier', ["M", "B"])

# Logistic Regression
print("\n--- Logistic Regression ---")
lg_clf = LogisticRegression(random_state=60, solver='liblinear')
lg_param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(lg_clf, lg_param_grid, cv=5)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, 'LogisticRegression', ["M", "B"])

# K-Nearest Neighbors
print("\n--- K-Nearest Neighbors ---")
knn_clf = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, 'KNeighborsClassifier', ["M", "B"])

# Support Vector Machine
print("\n--- Support Vector Machine ---")
svm_clf = SVC()
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm_clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
y_pred = grid_search.best_estimator_.predict(X_test)
print("Best parameters:", grid_search.best_params_)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(y_test, y_pred, 'SVC', ["M", "B"])
