# Detecting Cyber Threats in Medical Devices Using Machine Learning

This project presents a **Machine Learning-based Network Intrusion Detection System (ML-NIDS)** designed to detect cyber threats in medical devices during data transfers. It applies **Support Vector Machines (SVM)**, **k-Nearest Neighbors (KNN)**, and **Random Forest (RF)** to identify and classify intrusions in network traffic using the UNSW-NB15 dataset.

> Final Year B.Tech Project by **S. Praveen**  
Karunya Institute of Technology and Sciences, Coimbatore, India

---

## 🧠 Models Used

- **SVM (Support Vector Machine)**
- **KNN (k-Nearest Neighbors)**
- **Random Forest Classifier**

Each model is trained using One-vs-Rest classification and evaluated using accuracy, precision, recall, and ROC/PR curves.

---

## 📊 Dataset

The dataset used is **UNSW-NB15**, developed by the **Australian Centre for Cyber Security**. It contains a mix of benign and malicious network traffic with features extracted using Argus and Bro-IDS.

**Attack types included:**
- DoS (Denial of Service)
- U2R (User to Root)
- R2L (Remote to Local)
- Reconnaissance
- Fuzzers
- Worms
- Exploits

---

## ⚙️ Project Structure

```plaintext
intrusion-detection-ml/
│
├── data/
│   ├── training_data.csv
│   ├── test_data.csv
│   └── extra_data.xlsx
│
├── models/
│   ├── knn_model_ovr.sav
│   ├── svm_model_ovr.sav
│   ├── random_forest_model_ovr.sav
│   └── scaler_model.sav
│
├── scripts/
│   ├── knn_train.py
│   ├── svm_train.py
│   ├── rf_train.py
│   ├── knn_predict_gui.py
│   ├── svm_predict_gui.py
│   └── rf_predict_gui.py
│
├── results/
│   ├── knn_classification_report.txt
│   ├── svm_classification_report.txt
│   └── rf_classification_report.txt
│
├── images/
│   ├── knn_confusion_matrix.png
│   ├── svm_precision_recall_curve.png
│   ├── rf_roc_curve.png
│
├── README.md
├── requirements.txt
└── project_report.pdf

