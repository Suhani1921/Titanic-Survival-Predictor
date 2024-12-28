import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Step 1: Load the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Step 2: Data Preprocessing
# Handle missing values for Age and Embarked
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# Convert 'Sex' to numeric values
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# One-hot encode the 'Embarked' feature
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# Step 3: Select features and target variable
X = train[['Pclass', 'Sex', 'Age', 'Fare']]
y = train['Survived']
X_test = test[['Pclass', 'Sex', 'Age', 'Fare']]  # Make sure to select the same features as X

# Step 4: Train-Test Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Random Forest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate Model (on validation set)
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_val, y_pred))

# Step 7: Make Predictions on Test Data
y_test_pred = model.predict(X_test)

# Ensure the number of predictions matches the number of test data rows
print(len(y_test_pred), len(test))

# Step 8: Prepare Submission File
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # Make sure the PassengerId is from the test set
    'Survived': y_test_pred               # Predictions should have the same length
})

# Step 9: Save the submission
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
print('Accuracy:', accuracy_score(y_val, y_pred))

