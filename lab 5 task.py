# ============================================================
# Lab 05 - Treating a Classification Problem as a ML Problem
# Dataset: Titanic.csv
# ============================================================

# ---------------------- Import Required Libraries ----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import joblib

# ---------------------- Step 1: Data Loading ----------------------
df = pd.read_csv("tested.csv")
print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", df.shape)
print(df.head())

# ---------------------- Step 2: Data Exploration ----------------------
print("\n===== Basic Info =====")
print(df.info())
print("\n===== Missing Values =====")
print(df.isnull().sum())

sns.countplot(x='Survived', data=df)
plt.title('Survival Counts')
plt.show()

# ---------------------- Step 3: Feature Engineering ----------------------
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Fill missing values safely
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['SibSp'].fillna(df['SibSp'].median(), inplace=True)
df['Parch'].fillna(df['Parch'].median(), inplace=True)

# Drop irrelevant columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Final check — fill any remaining NaNs just in case
df.fillna(0, inplace=True)

print("\n✅ Missing Values After Cleaning:", df.isnull().sum().sum())

# ---------------------- Step 4: Split and Scale ----------------------
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'FamilySize']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\n✅ Data Splitting and Scaling Completed!")

# ---------------------- Step 5: Model Training ----------------------
models = {
    'LogisticRegression': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'SVC': SVC(probability=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# ---------------------- Step 6: ROC Curve (Best Model) ----------------------
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
y_probs = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {roc_auc_score(y_test, y_probs):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ---------------------- Step 7: Cross-Validation ----------------------
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\n✅ Cross-Validation Accuracy for {best_model_name}: {cv_scores.mean():.3f}")

# ---------------------- Step 8: Hyperparameter Tuning (Random Forest) ----------------------
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10]
}
rf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)

print("\nBest Parameters for RandomForest:", gs.best_params_)

# ---------------------- Step 9: Save the Best Model ----------------------
joblib.dump(gs.best_estimator_, 'titanic_best_model.joblib')
print("✅ Model saved successfully as 'titanic_best_model.joblib'")

# ---------------------- Step 10: Summary ----------------------
print("\n===== Model Performance Summary =====")
for model, acc in results.items():
    print(f"{model}: {acc:.3f}")
print(f"\n🏆 Best Model: {best_model_name} (Accuracy = {results[best_model_name]:.3f})")
