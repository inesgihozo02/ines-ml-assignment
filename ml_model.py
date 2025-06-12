import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Prepare features (X) and target (y)
# For Titanic, 'Survived' is the target
y = df['Survived']

# Select some features; here we'll use numeric and relevant columns
X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

# Handle missing values in features (simple fill with median)
X = X.fillna(X.median())

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
