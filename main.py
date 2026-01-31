# Mengimpor librari Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, classification_report, confusion_matrix


# Membaca data dari direktori komputer
df = pd.read_csv('Telco-Customer-Churn.csv')

# Menampilkan variabel dan ukuran data
print(df)
# Tinjau jumlah baris kolom dan jenis data dalam dataset dengan info.
print(df.info())
# Cek duplikasi
dup = df.duplicated().sum()

print(f"Jumlah duplikasi: {dup}")
# Mengidentifikasi missing values
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"{col} unique values: {df[col].unique()}")

print(df.describe())
# Cek tipe data TotalCharges
print("Tipe data TotalCharges:", df['TotalCharges'].dtype)

# Mengubah ke numerik untuk melihat error/null
# errors='coerce' akan mengubah text non-angka menjadi NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Cek apakah ada nilai null setelah konversi (ini berarti tadinya ada string kosong)
print("Jumlah missing values di TotalCharges:", df['TotalCharges'].isnull().sum())

# Menghitung jumlah Churn
churn_counts = df['Churn'].value_counts()

# Membuat Pie Chart
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title('Persentase Customer Churn')
# plt.show()

# Grafik Barchart
features = ['Contract', 'InternetService', 'PaymentMethod']

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, feature in enumerate(features):
    sns.countplot(x=feature, hue='Churn', data=df, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Churn berdasarkan {feature}')
    if feature == 'PaymentMethod':
        axes[i].tick_params(axis='x', rotation=45) # Memutar label agar tidak tabrakan

plt.tight_layout()
# plt.show()

# `*Membersihkan* data`
# 1. Menangani Kolom TotalCharges (Ubah ke numerik & isi yang kosong)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 2. Menangani Kolom Tenure (Isi yang kosong dengan median)
df['tenure'] = df['tenure'].fillna(df['tenure'].median())

# 3. Menangani Kolom Gender (Inkonsistensi M/F dan missing values)
df['gender'] = df['gender'].replace({'M': 'Male', 'F': 'Female'})
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# 4. Menangani Inkonsistensi Layanan (Penyederhanaan kategori)
cols_to_fix = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in cols_to_fix:
    df[col] = df[col].replace('No internet service', 'No')

# 5. Menghapus kolom customerID (Karena tidak memiliki nilai prediktif)
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Verifikasi Akhir: Cek apakah masih ada missing values
print("Pengecekan Akhir Missing Values:")
print(df.isnull().sum())
print("\nUkuran data sekarang:", df.shape)

# 1. Menentukan Label Data (Target)
# Mengubah Yes/No menjadi 1/0
df['Churn'] = np.where(df['Churn'] == 'Yes', 1, 0)

# 2. Menentukan Objek Data (Fitur) & Mengkonstruksi (One-Hot Encoding)
# Mengubah semua fitur kategorikal menjadi kolom angka binary (0/1)
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
y = df['Churn']

print("Konstruksi data selesai. Data siap dimasukkan ke model.")

# *Korelasi* Variabel
numerical_cols_with_churn = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']

# Heatmap antar variabel numerik
plt.figure(figsize=(10, 8))
sns.heatmap(df[numerical_cols_with_churn].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriks Korelasi Variabel Numerik termasuk Churn')
# plt.show()
print("Heatmap korelasi selesai.")

# **Membangun** Model

# Membagi data menjadi 80% Training dan 20% Testing
# Stratify memastikan proporsi churn tetap sama di kedua set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

# Model 2: Random Forest (Model Pilihan Lainnya)
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

print("Membangun model selesai. Siap untuk tahap evaluasi.")

# Membagi data menjadi 80% Training dan 20% Testing
# Stratify memastikan proporsi churn tetap sama di kedua set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_model.fit(X_train, y_train)

# Model 2: Random Forest (Model Pilihan Lainnya)
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

print("Membangun model selesai. Siap untuk tahap evaluasi.")

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

# Visualisasi
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Fitur Paling Berpengaruh terhadap Churn')
# plt.show()

# Evaluasi Model

# Inisialisasi model Decision Tree dengan max_depth = 3
model_dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)

# Latih model menggunakan data training
model_dt_pruned.fit(X_train, y_train)

# Lakukan prediksi pada data testing
y_pred_dt_pruned = model_dt_pruned.predict(X_test)

# Hitung metrik kinerja klasifikasi
accuracy_dt_pruned = accuracy_score(y_test, y_pred_dt_pruned)
precision_dt_pruned = precision_score(y_test, y_pred_dt_pruned)
recall_dt_pruned = recall_score(y_test, y_pred_dt_pruned)
f1_dt_pruned = f1_score(y_test, y_pred_dt_pruned)
fpr_dt_pruned, tpr_dt_pruned, thresholds_dt_pruned = roc_curve(y_test, model_dt_pruned.predict_proba(X_test)[:, 1])
auc_dt_pruned = auc(fpr_dt_pruned, tpr_dt_pruned)

# Tampilkan metrik kinerja
print("Metrik Kinerja Model Decision Tree (max_depth=3):")
print(f"Accuracy: {accuracy_dt_pruned:.4f}")
print(f"Precision: {precision_dt_pruned:.4f}")
print(f"Recall: {recall_dt_pruned:.4f}")
print(f"F1 Score: {f1_dt_pruned:.4f}")
print(f"AUC-ROC: {auc_dt_pruned:.4f}")

# Tampilkan classification report
print("\nClassification Report (max_depth=3):")
print(classification_report(y_test, y_pred_dt_pruned))

# Confusion Matrix
cm_dt_pruned = confusion_matrix(y_test, y_pred_dt_pruned)
plt.figure(figsize=(5,4))
sns.heatmap(cm_dt_pruned, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Decision Tree (max_depth=3)')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
# plt.show()

# Lakukan prediksi pada data testing untuk Random Forest
y_pred_rf = rf_model.predict(X_test)

# Hitung metrik kinerja klasifikasi untuk Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
auc_rf = auc(fpr_rf, tpr_rf)

# Tampilkan metrik kinerja
print("Metrik Kinerja Model Random Forest:")
print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1 Score: {f1_rf:.4f}")
print(f"AUC-ROC: {auc_rf:.4f}")

# Tampilkan classification report
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix: Random Forest')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.show()