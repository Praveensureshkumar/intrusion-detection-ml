import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, plot_precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
import joblib
import matplotlib.pyplot as plt

# Load your unbalanced dataset
data = pd.read_csv('Book3.csv')

# Separate features and labels
X = data.drop('A48', axis=1)
y = data['A48']

# Identify categorical columns (string) and numerical columns (float)
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['float64']).columns

# Preprocess categorical columns (e.g., apply Label Encoding)
for col in categorical_columns:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col])

# Initialize StandardScaler
scaler = StandardScaler()

# Preprocess numerical columns (e.g., apply Standardization)
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Save the trained scaler
joblib.dump(scaler, 'scaler_model.sav')

# Initialize SMOTE and Tomek Links
smote = SMOTE(sampling_strategy='minority')
tomek = TomekLinks(sampling_strategy='all')
smt = SMOTETomek(smote=smote, tomek=tomek)

# Apply SMOTE and TOMEK Links
X_resampled, y_resampled = smt.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train a KNN classifier with OneVsRest
knn_classifier_ovr = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1))
knn_classifier_ovr.fit(X_train, y_train)

# Make predictions on the test data
y_pred_ovr = knn_classifier_ovr.predict(X_test)

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
disp = plot_confusion_matrix(knn_classifier_ovr, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.figure_.suptitle("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# Plot and save the precision-recall curve for each class (OvR strategy)
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(rf_classifier_ovr.classes_)):
    plot_precision_recall_curve(rf_classifier_ovr.estimators_[i], X_test, (y_test == rf_classifier_ovr.classes_[i]), ax=ax, name=f'Class {i}')

plt.title("Precision-Recall Curve (OvR Strategy)")
plt.tight_layout()
plt.savefig("precision_recall_curve_ovr.png")
plt.show()



# Save the trained model to a .sav file
joblib.dump(knn_classifier_ovr, 'knn_model_ovr.sav')
