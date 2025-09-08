import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Define the file paths
train_file_path = "C:\\Users\\msraa\\cs440\\prediction_challenge_train.csv"
airport_map_file_path = "C:\\Users\\msraa\\cs440\\airport_country_code_mapping.csv"
test_file_path = "C:\\Users\\msraa\\cs440\\prediction_challenge_test.csv"

# Read the CSV files
train_df = pd.read_csv(train_file_path)
airport_map_df = pd.read_csv(airport_map_file_path)
test_df = pd.read_csv(test_file_path)

# Merge airport country mapping
train_df = train_df.merge(airport_map_df, on='Airport Country Code', how='left')
test_df = test_df.merge(airport_map_df, on='Airport Country Code', how='left')

# Feature Engineering
for df in [train_df, test_df]:
    df['Departure Date'] = pd.to_datetime(df['Departure Date'], errors='coerce')
    df['Departure Day'] = df['Departure Date'].dt.day
    df['Departure Month'] = df['Departure Date'].dt.month
    df['Departure Year'] = df['Departure Date'].dt.year
    df['Age Group'] = pd.cut(df['Age'], bins=[0, 12, 60, 100], labels=['Child', 'Adult', 'Senior'])

# Encoding categorical variables
train_df = pd.get_dummies(train_df, columns=['Gender', 'Age Group', 'Airport Country Code'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Gender', 'Age Group', 'Airport Country Code'], drop_first=True)

# Align test dataset with training dataset features
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

# Selecting features and target
features = ['Ticket Price', 'Departure Day', 'Departure Month', 'Departure Year'] + \
           [col for col in train_df.columns if col.startswith(('Gender_', 'Age Group_', 'Airport Country Code'))]
target = 'Eligible_For_Discount'

X_train = train_df[features]
y_train = (train_df[target] == 'Yes').astype(int)
X_test = test_df[features]

# Train/test split for evaluation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Hyperparameter tuning for better accuracy
param_grid = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_split, y_train_split)

# Best model
best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(X_val_split)

# Model Evaluation
accuracy = accuracy_score(y_val_split, y_val_pred)
conf_matrix = confusion_matrix(y_val_split, y_val_pred)
report = classification_report(y_val_split, y_val_pred)

# Decision Tree Visualization
decision_tree_rules = export_text(best_model, feature_names=list(X_train.columns))

# Predict on the test dataset
test_predictions = best_model.predict(X_test)

# Save predictions to the original test DataFrame
test_df['Eligible_For_Discount'] = ['Yes' if p == 1 else 'No' for p in test_predictions]

# Save the updated test DataFrame back to the CSV file
test_df.to_csv(test_file_path, index=False)

# Display results
print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)
print("Decision Tree Rules:\n", decision_tree_rules)