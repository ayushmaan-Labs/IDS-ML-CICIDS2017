import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the CICIDS2017 dataset (ensure you have the dataset in the same directory)
# Replace 'cicids2017.csv' with your actual path
data = pd.read_csv("cicids2017.csv")

# Preprocess the data
# Drop any columns with NaN or Inf values
data = data.replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA).dropna()

# Encode the labels (last column assumed to be 'Label')
label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

# Split features and labels
X = data.drop(columns=['Label'])
y = data['Label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train the Random Forest model
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save the model if needed (optional):
# import joblib
# joblib.dump(clf, "ids_model.joblib")
