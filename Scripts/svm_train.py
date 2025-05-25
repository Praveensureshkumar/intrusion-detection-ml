import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.multiclass import OneVsRestClassifier
import joblib
import matplotlib.pyplot as plt

# Load your unbalanced dataset
data = pd.read_csv('Book3.csv')

# Separate features and labels
X = data.drop('A48', axis=1)
y = data['A48']

# Identify numerical columns
numerical_columns = X.select_dtypes(include=['float64']).columns

# Filter out non-numeric columns (drop categorical columns)
X = X[numerical_columns]

# Preprocess numerical columns (e.g., apply Standardization)
X = StandardScaler().fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train an SVM classifier with OneVsRest
svm_classifier_ovr = OneVsRestClassifier(SVC(kernel='linear', C=1, random_state=42))
svm_classifier_ovr.fit(X_train, y_train)

# Make predictions on the test data
y_pred_ovr = svm_classifier_ovr.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred_ovr)
print(f'Accuracy: {accuracy}')

# Generate a classification report and save as a text file
class_report = classification_report(y_test, y_pred_ovr)
print("Classification Report:")
print(class_report)
with open('classification_report.txt', 'w') as f:
    f.write(class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_ovr)
print("Confusion Matrix:")
print(conf_matrix)

# Plot and save the confusion matrix
disp = ConfusionMatrixDisplay.from_estimator(svm_classifier_ovr, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.figure_.suptitle("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Plot and save the precision-recall curve for each class (OvR strategy)
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(svm_classifier_ovr.classes_)):
    PrecisionRecallDisplay.from_estimator(svm_classifier_ovr.estimators_[i], X_test, (y_test == svm_classifier_ovr.classes_[i]), ax=ax, name=f'Class {i}')

plt.title("Precision-Recall Curve (OvR Strategy)")
plt.tight_layout()
plt.savefig("precision_recall_curve_ovr.png")

# Plot and save the ROC curve for each class (OvR strategy)
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(svm_classifier_ovr.classes_)):
    RocCurveDisplay.from_estimator(svm_classifier_ovr.estimators_[i], X_test, (y_test == svm_classifier_ovr.classes_[i]), ax=ax, name=f'Class {i}')

plt.title("ROC Curve (OvR Strategy)")
plt.tight_layout()
plt.savefig("roc_curve_ovr.png")

# Save the trained model to a .sav file
joblib.dump(svm_classifier_ovr, 'svm_model_ovr.sav')
