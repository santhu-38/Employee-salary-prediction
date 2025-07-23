# salary_prediction.py


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

# Load and clean data
data = pd.read_csv("C:/Users/pitha/OneDrive/Desktop/EmployeeSalaryPrediction/adult 3.csv", encoding="latin1", on_bad_lines="skip")

#observe the data
data.count()
data.isna()
data.isna().count()
data.isna().sum()
data.describe()
data["gender"].unique()
data.gender.value_counts()
data.workclass.value_counts()
data.occupation.value_counts()
data.dropna(inplace=True) # if null values are present


# Basic cleaning
data = data[data["occupation"] != "Armed-Forces"]
data = data[~data['workclass'].isin(["Without-pay", "Never-worked"])]
data = data[~data['education'].isin(["Preschool", "1st-4th", "5th-6th"])]
data['workclass'].replace("?", "others", inplace=True)
data['occupation'].replace("?", "others", inplace=True)
plt.boxplot(data['age'])
plt.show()
data = data[(data['age'] >= 17) & (data['age'] <= 75)]
data = data[~data['workclass'].isin(["Self-emp-not-inc", "Self-emp-inc"])]
data.drop(columns=["education", "fnlwgt", "race", "marital-status", "relationship", "native-country"], inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_workclass = LabelEncoder()
le_occupation = LabelEncoder()

data['gender'] = le_gender.fit_transform(data['gender'])
data['workclass'] = le_workclass.fit_transform(data['workclass'])
data['occupation'] = le_occupation.fit_transform(data['occupation'])

# Save encoders
joblib.dump(le_gender, 'le_gender.pkl')
joblib.dump(le_workclass, 'le_workclass.pkl')
joblib.dump(le_occupation, 'le_occupation.pkl')

# Encode target labels
X = data.drop(columns=['income'])
y = data['income']
y = y.map({'<=50K': 0, '>50K': 1})

# Split data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)

# Train pipeline with RandomForest
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(class_weight='balanced'),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
best_model = None
best_accuracy = 0.0
best_model_name = ""

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(xtrain, ytrain)
    y_pred = pipe.predict(xtest)
    acc = accuracy_score(ytest, y_pred)
    results[name] = acc

    print(f"\n🔍 Model: {name}")
    print(f"✅ Accuracy: {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(ytest, y_pred))
    print("🧮 Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred))

    if acc > best_accuracy:
        best_model = pipe
        best_accuracy = acc
        best_model_name = name

# Save best model
joblib.dump(best_model, 'salary_model.pkl')
print(f"\n✅ Best Model Saved: {best_model_name} with Accuracy: {best_accuracy:.4f}")
