# ============================================================
# HR ANALYTICS - EMPLOYEE ATTRITION PREDICTION
# Internship Project
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("charts", exist_ok=True)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print("HR EMPLOYEE ATTRITION ANALYSIS")
print("=" * 60)

# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("\nDuplicates removed.")
print("Current rows:", len(df))

# ============================================================
# 5. ATTRITION SUMMARY
# ============================================================

attrition_count = df["Attrition"].value_counts()

print("\nAttrition Count:")
print(attrition_count)

attrition_percentage = df["Attrition"].value_counts(normalize=True) * 100

print("\nAttrition Percentage:")
print(attrition_percentage.round(2))

# ============================================================
# 6. KPI VALUES
# ============================================================

total_employees = len(df)
employees_left = (df["Attrition"] == "Yes").sum()
employees_stayed = (df["Attrition"] == "No").sum()

attrition_rate = (employees_left / total_employees) * 100

print("\n" + "=" * 60)
print("KEY HR KPIs")
print("=" * 60)

print("Total Employees:", total_employees)
print("Employees Left:", employees_left)
print("Employees Stayed:", employees_stayed)
print("Attrition Rate:", round(attrition_rate, 2), "%")

# ============================================================
# 7. CHART 1 - ATTRITION DISTRIBUTION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Attrition")

plt.title("Employee Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("charts/01_attrition_distribution.png", dpi=300)
plt.close()

# ============================================================
# 8. CHART 2 - DEPARTMENT-WISE ATTRITION
# ============================================================

department_attrition = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nDepartment-wise Attrition Percentage:")
print(department_attrition.round(2))

department_attrition.plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Department-wise Attrition Percentage")
plt.xlabel("Department")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Attrition")

plt.tight_layout()
plt.savefig("charts/02_department_attrition.png", dpi=300)
plt.close()

# ============================================================
# 9. CHART 3 - JOB ROLE-WISE ATTRITION
# ============================================================

jobrole_attrition = pd.crosstab(
    df["JobRole"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nJob Role-wise Attrition Percentage:")
print(jobrole_attrition.round(2))

jobrole_attrition.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Job Role-wise Attrition Percentage")
plt.xlabel("Job Role")
plt.ylabel("Percentage")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Attrition")

plt.tight_layout()
plt.savefig("charts/03_jobrole_attrition.png", dpi=300)
plt.close()

# ============================================================
# 10. CHART 4 - OVERTIME VS ATTRITION
# ============================================================

overtime_attrition = pd.crosstab(
    df["OverTime"],
    df["Attrition"],
    normalize="index"
) * 100

print("\nOvertime vs Attrition:")
print(overtime_attrition.round(2))

overtime_attrition.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Overtime vs Employee Attrition")
plt.xlabel("Overtime")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Attrition")

plt.tight_layout()
plt.savefig("charts/04_overtime_attrition.png", dpi=300)
plt.close()

# ============================================================
# 11. CHART 5 - AGE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Age",
    hue="Attrition",
    bins=20,
    kde=True
)

plt.title("Age Distribution by Attrition")
plt.xlabel("Age")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig("charts/05_age_attrition.png", dpi=300)
plt.close()

# ============================================================
# 12. CHART 6 - MONTHLY INCOME VS ATTRITION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.tight_layout()
plt.savefig("charts/06_income_attrition.png", dpi=300)
plt.close()

# ============================================================
# 13. CHART 7 - YEARS AT COMPANY VS ATTRITION
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="YearsAtCompany"
)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Years at Company")

plt.tight_layout()
plt.savefig("charts/07_years_company_attrition.png", dpi=300)
plt.close()

# ============================================================
# 14. PREPARE DATA FOR MACHINE LEARNING
# ============================================================

ml_data = df.copy()

# Convert Attrition to numeric
ml_data["Attrition"] = ml_data["Attrition"].map({
    "Yes": 1,
    "No": 0
})

# Remove columns that are not useful for prediction
columns_to_remove = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

ml_data = ml_data.drop(
    columns=columns_to_remove,
    errors="ignore"
)

# Convert categorical columns into dummy variables
ml_data = pd.get_dummies(
    ml_data,
    drop_first=True
)

# ============================================================
# 15. SEPARATE FEATURES AND TARGET
# ============================================================

X = ml_data.drop("Attrition", axis=1)
y = ml_data["Attrition"]

print("\nMachine Learning Features:", X.shape[1])

# ============================================================
# 16. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ============================================================
# 17. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 18. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression model trained successfully!")

# ============================================================
# 19. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test_scaled)

# ============================================================
# 20. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title("Employee Attrition Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("charts/08_confusion_matrix.png", dpi=300)
plt.close()

# ============================================================
# 22. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Importance"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nTop 15 Important Factors:")
print(
    feature_importance[
        ["Feature", "Importance"]
    ].head(15)
)

# Save feature importance
feature_importance.to_csv(
    "output/feature_importance.csv",
    index=False
)

# ============================================================
# 23. FEATURE IMPORTANCE CHART
# ============================================================

top_features = feature_importance.head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Factors Related to Employee Attrition")
plt.xlabel("Model Coefficient")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("charts/09_feature_importance.png", dpi=300)
plt.close()

# ============================================================
# 24. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "output/employee_attrition_model.pkl"
)

joblib.dump(
    scaler,
    "output/scaler.pkl"
)

print("\nModel saved successfully.")

# ============================================================
# 25. CREATE POWER BI DATASET
# ============================================================

powerbi_data = df.copy()

# Add predicted attrition for all employees
all_features = ml_data.drop("Attrition", axis=1)

all_features_scaled = scaler.transform(all_features)

all_predictions = model.predict(all_features_scaled)

all_probabilities = model.predict_proba(
    all_features_scaled
)[:, 1]

powerbi_data["Predicted_Attrition"] = np.where(
    all_predictions == 1,
    "High Risk",
    "Low Risk"
)

powerbi_data["Attrition_Risk_Percentage"] = (
    all_probabilities * 100
).round(2)

# ============================================================
# 26. CREATE RISK CATEGORY
# ============================================================

def risk_category(probability):

    if probability >= 0.70:
        return "High Risk"

    elif probability >= 0.40:
        return "Medium Risk"

    else:
        return "Low Risk"


powerbi_data["Risk_Category"] = [
    risk_category(x)
    for x in all_probabilities
]

# ============================================================
# 27. SAVE POWER BI DATA
# ============================================================

powerbi_data.to_csv(
    "output/HR_Attrition_PowerBI_Data.csv",
    index=False
)

print("\nPower BI dataset created successfully.")

# ============================================================
# 28. SAVE MODEL RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        round(accuracy * 100, 2),
        round(precision * 100, 2),
        round(recall * 100, 2),
        round(f1 * 100, 2)
    ]
})

results.to_csv(
    "output/model_performance.csv",
    index=False
)

# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT PROCESS COMPLETED")
print("=" * 60)

print("\nTotal Employees:", total_employees)
print("Employees Left:", employees_left)
print("Attrition Rate:", round(attrition_rate, 2), "%")

print("\nModel Results:")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nFiles created:")
print("1. charts/")
print("2. output/HR_Attrition_PowerBI_Data.csv")
print("3. output/feature_importance.csv")
print("4. output/model_performance.csv")
print("5. output/employee_attrition_model.pkl")
print("6. output/scaler.pkl")

print("\nHR Attrition Project completed successfully!")