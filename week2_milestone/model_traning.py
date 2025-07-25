import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
file_path = r"C:\Users\HP\OneDrive\Desktop\greenhouse emission\week1_milestone\SupplyChainEmissionFactorsforUSIndustriesCommodities2015_Summary (2).csv"
df = pd.read_csv(file_path)

# Clean-up
df.drop(columns=['Unnamed: 7'], errors='ignore', inplace=True)
df.dropna(inplace=True)
target_column = 'Supply Chain Emission Factors without Margins'
df = df[pd.to_numeric(df[target_column], errors='coerce').notnull()]
df[target_column] = df[target_column].astype(float)

# Encode categorical features
X = pd.get_dummies(df.drop(columns=[target_column]), drop_first=True)
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------------------------
#  Train 3 Models and Compare (for RMSE Plot)
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    
    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R² Score": r2
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# -----------------------------------------------
#  RMSE Comparison Plot (shown at the top)
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='RMSE', palette='magma')
plt.title("Model Comparison – RMSE")
plt.ylabel("Root Mean Squared Error")
plt.xlabel("Model")
plt.tight_layout()
plt.show()

# Print the table for clarity
print(results_df)

# -----------------------------------------------
#  Hyperparameter Tuning for Best RF
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_pred = best_model.predict(X_test)

# Eval best model
rmse = np.sqrt(mean_squared_error(y_test, best_pred))
r2 = r2_score(y_test, best_pred)

print(f"\n Tuned RF RMSE: {rmse:.4f}")
print(f" Tuned RF R²: {r2:.4f}")

# -----------------------------------------------
#  Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, best_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.title("Actual vs Predicted  Tuned RF")
plt.tight_layout()
plt.show()

# -----------------------------------------------
#  Feature Importance
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Top 15 Feature Importances ")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Distribution of Target
plt.figure(figsize=(10, 5))
sns.histplot(y, kde=True, bins=30, color='skyblue')
plt.title("Distribution of Target Emissions")
plt.xlabel("Emission Value")
plt.tight_layout()
plt.show()



importances = best_model.feature_importances_
feat_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_importance_df, palette="magma")
plt.title("Top 10 Feature Importances – Random Forest")
plt.tight_layout()
plt.show()


















