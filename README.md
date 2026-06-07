# Kalp Hastalığı Tahmini - Karar Ağacı Sınıflandırma

BLM0463 Veri Madenciliğine Giriş dersi dönem projesidir. UCI Cleveland Heart Disease veri seti üzerinde Karar Ağacı algoritması ile kalp hastalığı tahmini gerçekleştirilmiştir.

## Sunum Videosu

[YouTube - Proje Sunum Videosu](https://www.youtube.com/watch?v=PS37XqIn5VY)

## Veri Seti

- **Kaynak:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Örnek sayısı:** 303
- **Özellik sayısı:** 13 + 1 hedef değişken
- **Sınıflar:** Sağlıklı (164) / Hasta (139)

## Yöntem

- **Algoritma:** Decision Tree (Karar Ağacı)
- **Bölme kriteri:** Gini Impurity
- **Eğitim/Test oranı:** %80 / %20 (Stratified)
- **Çapraz doğrulama:** 10-Katlamalı Stratified K-Fold

## Sonuçlar

| Metrik | Değer |
|--------|-------|
| Accuracy | %80.33 |
| Precision | %76.67 |
| Sensitivity | %82.14 |
| Specificity | %78.79 |
| F1-Score | %79.31 |
| AUC-ROC | 0.8528 |


## Referans

Hassan, C.A.u., Iqbal, J., Irfan, R., Hussain, S., Algarni, A.D., Bukhari, S.S.H., Alturki, N., Ullah, S.S. (2022). *Effectively Predicting the Presence of Coronary Heart Disease Using Machine Learning Classifiers.* Sensors, 22(19), 7227. DOI: 10.3390/s22197227
