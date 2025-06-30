# Part 1: Loading and Inspecting the Data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing datasets
try:
    train_df = pd.read_csv('Train_Data.csv')
    test_df = pd.read_csv('Test_Data.csv')
except FileNotFoundError:
    print("Ensure 'Train_Data.csv' and 'Test_Data.csv' are in the same directory.")
    exit()

# Display the first few rows of the training DataFrame
print("First 5 rows of the training dataset:")
print(train_df.head())
print("\n")

# Display the first few rows of the testing DataFrame
print("First 5 rows of the testing dataset:")
print(test_df.head())
print("\n")

# Get a summary of the training DataFrame
print("Training DataFrame info:")
print(train_df.info())
print("\n")

# Get a summary of the testing DataFrame
print("Testing DataFrame info:")
print(test_df.info())
print("\n")


# Part 2: Data Preprocessing

# Store 'SEQN' from the test_df for the final submission
test_seqn = test_df['SEQN']

# Drop the 'SEQN' column from both training and testing data as it's an identifier
train_df = train_df.drop('SEQN', axis=1)
test_df = test_df.drop('SEQN', axis=1)

# Ensure 'age_group' in train_df is integer type (0 or 1)
if 'age_group' in train_df.columns:
    train_df['age_group'] = train_df['age_group'].astype(int)
else:
    print("Error: 'age_group' column not found in training data.")
    exit()

# Define numerical features for imputation
numerical_features = ['PAQ605', 'BMXBMI', 'LBXGLU', 'DIQ010', 'LBXGLT', 'LBXIN']

# Impute missing values in both training and testing datasets with the median
# It's crucial to calculate medians from the training data to avoid data leakage
for column in numerical_features:
    if train_df[column].isnull().sum() > 0:
        median_val_train = train_df[column].median()
        train_df[column].fillna(median_val_train, inplace=True)
        # Use the median calculated from the training data for the test set
        test_df[column].fillna(median_val_train, inplace=True)

print("Missing values in training data after imputation:")
print(train_df.isnull().sum())
print("\n")

print("Missing values in testing data after imputation:")
print(test_df.isnull().sum())
print("\n")

# Verify that all features in test_df are present in train_df and vice-versa
train_features = set(train_df.drop('age_group', axis=1).columns)
test_features = set(test_df.columns)

if train_features != test_features:
    print("Mismatch in features between training and testing datasets!")
    print("Features in training but not in test:", train_features - test_features)
    print("Features in test but not in training:", test_features - train_features)
    # Handle this based on requirements, for now, we assume they match after dropping SEQN
    # If there are mismatches, you might need to align columns or drop extra ones.
print("\n")


# Part 3: Splitting the Data (for training data validation, if needed)
# In Kaggle notebooks, you often train on the entire 'Train_Data.csv' and predict on 'Test_Data.csv'.
# However, for local validation, you might split the train_df further.
# For this notebook, we will use the entire train_df for training.

# Define the features (X_train_full) and the target (y_train_full) for full training
X_train_full = train_df.drop('age_group', axis=1)
y_train_full = train_df['age_group']

# Align columns - crucial if test_df had different column order or extra columns after dropping SEQN
# This ensures that the order of columns in X_test is the same as X_train_full
X_test_aligned = test_df[X_train_full.columns]


print("Shape of full training features (X_train_full):", X_train_full.shape)
print("Shape of full training target (y_train_full):", y_train_full.shape)
print("Shape of aligned test features (X_test_aligned):", X_test_aligned.shape)
print("\n")


# Part 4: Model Training

# Initialize the Random Forest Classifier
# A random_state is set for reproducibility
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the full training data
print("Training the model on the full training data...")
model.fit(X_train_full, y_train_full)
print("Model training complete.")
print("\n")


# Part 5: Model Evaluation (on a validation set, if created) or directly for submission predictions

# In a typical Kaggle scenario, you'd make predictions on the provided test set
# and then generate the submission file.
# If you want to evaluate your model's performance on unseen data before submitting,
# you would split `train_df` into train/validation sets in Part 3.

# For this example, we will directly predict on X_test_aligned (the provided test_df)

# Make predictions on the preprocessed test set
test_predictions = model.predict(X_test_aligned)

print("Predictions made on the test dataset.")
print("\n")


# Part 6: Generate Submission File

# Create a DataFrame for submission
# It should contain 'SEQN' and your predicted 'age_group'
submission_df = pd.DataFrame({'SEQN': test_seqn, 'age_group': test_predictions})

# Save the submission DataFrame to a CSV file
submission_file_name = 'submission.csv'
submission_df.to_csv(submission_file_name, index=False)

print(f"Submission file '{submission_file_name}' created successfully.")
print("First 5 rows of the submission file:")
print(submission_df.head())

# Optional: Feature Importance Plot (using the full training data)
feature_importances = model.feature_importances_
feature_names = X_train_full.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance_df)
plt.title('Feature Importances from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show() # Display the plot if running locally
