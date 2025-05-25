import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, plot_precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
import joblib
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def preprocess_data(data, categorical_columns, numerical_columns, scaler):
    # Preprocess categorical columns
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])

    # Preprocess numerical columns
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    return data

def make_predictions():
    try:
        # Load the test dataset
        test_data = pd.read_csv('test.csv')

        # Separate features and labels
        X_test_data = test_data.drop('A48', axis=1)
        y_test_data = test_data['A48']

        # Identify categorical columns (string) and numerical columns (float)
        categorical_columns_test = X_test_data.select_dtypes(include=['object']).columns
        numerical_columns_test = X_test_data.select_dtypes(include=['float64']).columns

        # Load the scaler from the training phase
        scaler = StandardScaler()
        scaler = joblib.load('scaler_model.sav')  # Assuming you saved the scaler during training

        # Preprocess the test data
        X_test_data = preprocess_data(X_test_data, categorical_columns_test, numerical_columns_test, scaler)

        # Load the trained OneVsRest Random Forest model
        rf_classifier_ovr = joblib.load('random_forest_model_ovr.sav')

        # Make predictions on the test data
        y_pred_test = rf_classifier_ovr.predict(X_test_data)

        # Display results in Tkinter window
        result_window = tk.Tk()
        result_window.title("Test Results")

        ttk.Label(result_window, text="Predicted Labels:").pack(pady=10)

        # Convert predicted labels to a string for display
        predicted_labels_str = ', '.join(map(str, y_pred_test))

        ttk.Label(result_window, text=predicted_labels_str).pack(pady=10)

        # Create a figure for the confusion matrix
        


        result_window.mainloop()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create Tkinter window
root = tk.Tk()
root.title("Test Model GUI")

# Button to trigger predictions
predict_button = tk.Button(root, text="Make Predictions", command=make_predictions)
predict_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
