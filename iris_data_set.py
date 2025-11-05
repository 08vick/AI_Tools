#Classical ML with Scikit-learn
#Dataset: Iris Species
#Goal: Preprocess, train Decision Tree, evaluate


# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np

# Step 1: Load the Iris dataset
# The Iris dataset is built into scikit-learn and contains 150 samples of iris flowers
# with 4 features (sepal length, sepal width, petal length, petal width) and 3 species.
iris = load_iris()
X = iris.data  # Feature matrix (150 x 4)
y = iris.target  # Target labels (0: setosa, 1: versicolor, 2: virginica)

# Convert to DataFrame for easier inspection
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y

# Step 2: Preprocessing
# Handle missing values
# Although the Iris dataset has no missing values, we check for completeness.
print("Checking for missing values:")
print(df.isnull().sum())


# Encode labels
# In this dataset, labels are already numeric (0, 1, 2), so no encoding is needed.
# However, if labels were strings e.g., 'setosa', we would use LabelEncoder:
#   from sklearn.preprocessing import LabelEncoder
#   le = LabelEncoder()
#   y = le.fit_transform(y_string_labels)
# Since y is already encoded, we proceed with it as-is.


# Step 3: Split the data
# Split into training (80%) and testing (20%) sets, stratified by class
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Ensures equal representation of each species in train/test
)

# Step 4: Train the model
# Initialize a Decision Tree Classifier with a fixed random state for reproducibility
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = clf.predict(X_test)

# Step 6: Evaluate the model
# Compute key metrics
accuracy = accuracy_score(y_test, y_pred)
# Use 'weighted' average for multi-class precision and recall (accounts for class imbalance)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Print results
print(f"\nModel Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))