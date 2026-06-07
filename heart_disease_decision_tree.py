"""
BLM0463 Veri Madenciliğine Giriş - Proje
Heart Disease Prediction using Decision Tree
Dataset: UCI Heart Disease (Cleveland)
Author: Yunus Ayaz - 23360859455
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')
import os

# Çıktı klasörü
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. VERİ SETİ YÜKLEME
# ============================================================
print("=" * 60)
print("1. VERİ SETİ YÜKLEME")
print("=" * 60)

# Veri setini yerel dosyadan veya UCI'dan yükle
data_path = "heart+disease/processed.cleveland.data"
if not os.path.exists(data_path):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    data_path = url

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv(data_path, names=columns, na_values='?')

print(f"Veri seti boyutu: {df.shape}")
print(f"\nİlk 5 satır:")
print(df.head())
print(f"\nVeri seti istatistikleri:")
print(df.describe())

# ============================================================
# 2. VERİ ÖN İŞLEME
# ============================================================
print("\n" + "=" * 60)
print("2. VERİ ÖN İŞLEME")
print("=" * 60)

# Eksik veri analizi
print(f"\nEksik veri sayıları:")
print(df.isnull().sum())

# Eksik verileri medyan ile doldur
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df.fillna(df.median(), inplace=True)

# Target: 0 = sağlıklı, 1 = hasta (orijinalde 0-4, biz binary yapıyoruz)
df['target'] = (df['target'] > 0).astype(int)

print(f"\nSınıf dağılımı:")
print(df['target'].value_counts())
print(f"  0 (Sağlıklı): {(df['target'] == 0).sum()}")
print(f"  1 (Hasta):    {(df['target'] == 1).sum()}")

# ============================================================
# 3. KEŞİFSEL VERİ ANALİZİ (EDA) GÖRSELLERİ
# ============================================================
print("\n" + "=" * 60)
print("3. KEŞİFSEL VERİ ANALİZİ")
print("=" * 60)

# 3a. Sınıf dağılımı
fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#2ecc71', '#e74c3c']
df['target'].value_counts().plot(kind='bar', color=colors, ax=ax, edgecolor='black')
ax.set_xticklabels(['Sağlıklı (0)', 'Hasta (1)'], rotation=0)
ax.set_ylabel('Sayı')
ax.set_title('Hedef Değişken Dağılımı')
for i, v in enumerate(df['target'].value_counts().values):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/sinif_dagilimi.png', dpi=150, bbox_inches='tight')
plt.close()

# 3b. Korelasyon matrisi
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, square=True)
ax.set_title('Korelasyon Matrisi', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/korelasyon_matrisi.png', dpi=150, bbox_inches='tight')
plt.close()

# 3c. Yaş dağılımı
fig, ax = plt.subplots(figsize=(8, 5))
df[df['target'] == 0]['age'].hist(alpha=0.6, label='Sağlıklı', color='#2ecc71', bins=20, ax=ax, edgecolor='black')
df[df['target'] == 1]['age'].hist(alpha=0.6, label='Hasta', color='#e74c3c', bins=20, ax=ax, edgecolor='black')
ax.set_xlabel('Yaş')
ax.set_ylabel('Frekans')
ax.set_title('Yaşa Göre Kalp Hastalığı Dağılımı')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/yas_dagilimi.png', dpi=150, bbox_inches='tight')
plt.close()

# 3d. Özellik kutu grafikleri
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, feat in enumerate(numeric_features):
    sns.boxplot(x='target', y=feat, data=df, ax=axes[i], palette=colors)
    axes[i].set_xticklabels(['Sağlıklı', 'Hasta'])
    axes[i].set_title(feat)
plt.suptitle('Sayısal Özelliklerin Sınıflara Göre Dağılımı', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/kutu_grafikleri.png', dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# 4. MODEL EĞİTİMİ
# ============================================================
print("\n" + "=" * 60)
print("4. DECISION TREE MODELİ EĞİTİMİ")
print("=" * 60)

X = df.drop('target', axis=1)
y = df['target']

# Train-Test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Eğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti:   {X_test.shape[0]} örnek")

# Decision Tree - Optimum parametrelerle
dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
y_prob = dt_model.predict_proba(X_test)[:, 1]

# ============================================================
# 5. DEĞERLENDİRME METRİKLERİ
# ============================================================
print("\n" + "=" * 60)
print("5. DEĞERLENDİRME METRİKLERİ")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)  # Sensitivity
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print(f"\nAccuracy:    {acc:.4f}  ({acc * 100:.2f}%)")
print(f"Precision:   {prec:.4f}  ({prec * 100:.2f}%)")
print(f"Recall/Sensitivity: {rec:.4f}  ({rec * 100:.2f}%)")
print(f"Specificity: {specificity:.4f}  ({specificity * 100:.2f}%)")
print(f"F1-Score:    {f1:.4f}  ({f1 * 100:.2f}%)")

print(f"\nConfusion Matrix:")
print(cm)
print(f"\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")

print(f"\nDetaylı Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=['Sağlıklı', 'Hasta']))

# ============================================================
# 6. ÇAPRAZ DOĞRULAMA (Cross-Validation)
# ============================================================
print("\n" + "=" * 60)
print("6. 10-KATLAMALI ÇAPRAZ DOĞRULAMA")
print("=" * 60)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(dt_model, X, y, cv=cv, scoring='accuracy')

print(f"\nHer katlamanın doğruluk değeri:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Katlama {i:2d}: {score:.4f}")
print(f"\nOrtalama Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

cv_precision = cross_val_score(dt_model, X, y, cv=cv, scoring='precision')
cv_recall = cross_val_score(dt_model, X, y, cv=cv, scoring='recall')
cv_f1 = cross_val_score(dt_model, X, y, cv=cv, scoring='f1')

print(f"Ortalama Precision: {cv_precision.mean():.4f}")
print(f"Ortalama Recall:    {cv_recall.mean():.4f}")
print(f"Ortalama F1-Score:  {cv_f1.mean():.4f}")

# ============================================================
# 7. GÖRSELLEŞTİRMELER
# ============================================================
print("\n" + "=" * 60)
print("7. GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
print("=" * 60)

# 7a. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Sağlıklı (0)', 'Hasta (1)'],
            yticklabels=['Sağlıklı (0)', 'Hasta (1)'],
            annot_kws={'size': 16})
ax.set_xlabel('Tahmin Edilen', fontsize=12)
ax.set_ylabel('Gerçek', fontsize=12)
ax.set_title('Karışıklık Matrisi (Confusion Matrix)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Confusion Matrix kaydedildi")

# 7b. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'Decision Tree (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Rastgele Sınıflandırıcı')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.set_title('ROC Eğrisi')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - ROC Curve kaydedildi")

# 7c. Precision-Recall Curve
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(rec_curve, prec_curve)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec_curve, prec_curve, color='#3498db', lw=2, label=f'Decision Tree (AUC = {pr_auc:.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Eğrisi')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Precision-Recall Curve kaydedildi")

# 7d. Decision Tree Visualization
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(dt_model, feature_names=X.columns.tolist(),
          class_names=['Sağlıklı', 'Hasta'],
          filled=True, rounded=True, ax=ax,
          fontsize=8, proportion=True)
ax.set_title('Karar Ağacı Görselleştirmesi', fontsize=16)
plt.tight_layout()
plt.savefig(f'{output_dir}/decision_tree.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Decision Tree görselleştirmesi kaydedildi")

# 7e. Feature Importance
importances = dt_model.feature_importances_
feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#3498db', edgecolor='black')
ax.set_xlabel('Önem Derecesi')
ax.set_title('Özellik Önem Dereceleri (Feature Importance)')
for i, v in enumerate(feat_imp['Importance'].values):
    if v > 0:
        ax.text(v + 0.005, i, f'{v:.3f}', va='center')
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Feature Importance kaydedildi")

# 7f. Cross-Validation Scores Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f'K{i}' for i in range(1, 11)]
bars = ax.bar(folds, cv_scores, color='#9b59b6', edgecolor='black', alpha=0.8)
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', lw=2,
           label=f'Ortalama: {cv_scores.mean():.4f}')
ax.set_xlabel('Katlama')
ax.set_ylabel('Accuracy')
ax.set_title('10-Katlamalı Çapraz Doğrulama Sonuçları')
ax.legend()
ax.set_ylim(0.5, 1.05)
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{score:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{output_dir}/cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Cross-Validation sonuçları kaydedildi")

# 7g. Metrikler karşılaştırma bar chart
fig, ax = plt.subplots(figsize=(8, 5))
metrics_names = ['Accuracy', 'Precision', 'Recall\n(Sensitivity)', 'Specificity', 'F1-Score']
metrics_vals = [acc, prec, rec, specificity, f1]
colors_m = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax.bar(metrics_names, metrics_vals, color=colors_m, edgecolor='black', width=0.6)
for bar, val in zip(bars, metrics_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', fontweight='bold')
ax.set_ylim(0, 1.15)
ax.set_ylabel('Değer')
ax.set_title('Decision Tree Değerlendirme Metrikleri')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/metrikler.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - Metrikler karşılaştırması kaydedildi")

# ============================================================
# 8. FARKLI DERİNLİK ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("8. FARKLI DERİNLİK ANALİZİ")
print("=" * 60)

depths = range(1, 16)
train_accs = []
test_accs = []

for d in depths:
    dt_temp = DecisionTreeClassifier(max_depth=d, random_state=42,
                                     min_samples_split=5, min_samples_leaf=3)
    dt_temp.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt_temp.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt_temp.predict(X_test)))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(depths, train_accs, 'o-', color='#3498db', label='Eğitim Doğruluğu', lw=2)
ax.plot(depths, test_accs, 's-', color='#e74c3c', label='Test Doğruluğu', lw=2)
ax.set_xlabel('Ağaç Derinliği (max_depth)')
ax.set_ylabel('Accuracy')
ax.set_title('Ağaç Derinliğine Göre Model Performansı')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(list(depths))
plt.tight_layout()
plt.savefig(f'{output_dir}/derinlik_analizi.png', dpi=150, bbox_inches='tight')
plt.close()

best_depth = list(depths)[np.argmax(test_accs)]
print(f"En iyi test accuracy: {max(test_accs):.4f} (depth={best_depth})")

# ============================================================
# 9. SONUÇ ÖZETİ
# ============================================================
print("\n" + "=" * 60)
print("9. SONUÇ ÖZETİ")
print("=" * 60)

print(f"""
====================================
DECISION TREE MODEL SONUÇLARI
====================================
Test Seti Sonuçları:
  Accuracy:     {acc * 100:.2f}%
  Precision:    {prec * 100:.2f}%
  Sensitivity:  {rec * 100:.2f}%
  Specificity:  {specificity * 100:.2f}%
  F1-Score:     {f1 * 100:.2f}%
  AUC-ROC:      {roc_auc:.4f}

10-Fold Cross-Validation:
  Ortalama Accuracy:  {cv_scores.mean() * 100:.2f}%
  Std:                {cv_scores.std() * 100:.2f}%

Confusion Matrix:
  TP={tp}, TN={tn}, FP={fp}, FN={fn}

Karar Ağacı Parametreleri:
  criterion:        gini
  max_depth:        5
  min_samples_split: 5
  min_samples_leaf:  3
====================================
""")

print("Tüm görseller ve metrikler kaydedildi!")
print(f"Çıktı dizini: {output_dir}/")
