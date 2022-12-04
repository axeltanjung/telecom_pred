# Deployment Model Machine Learning
## Prediksi Customer Churn pada Perusahaan Telekomunikasi

## Latar Belakang

Dengan perkembangan telekomunikasi industry yang cepat, service provider semakin menurun akibat ekspansi dari ke arah yang berbasis subscriber. Untuk memenuhi kebutuhan akan perusahaan atas lingkungan yang kompetitif, retensi dari customer eksisting menjadi tantangan yang besar. Hal ini menyatakan bahwa cost untuk mendapatkan customer baru lebih tinggi dibandingkan mempertahankan yang lama. 

Maka dari itu, sebuah keniscayaan untuk industri telekomunikasi menggunakan advance analysis untuk memahami perilaku consumer dan hal tersebut dapat memprediksi asosiasi dari customer terhadap apakah customer tersebut akan meninggalkan layanan dari perusahaan. Dengan dilakukannya langkah prediktif tersebut, maka perusahaan telekomunikasi dapat menyusun langkah strategis terhadap segmentasi dari customer sehingga dapat di implementasikan strategi yang mengedepankan solusi terhadap permasalahan customer churn tersebut. 

## Objective
* Tujuan utama dari task ini adalah untuk memprediksi churn dari customer telekomunikasi berdasarkan beberapa variabel.
* Metode yang berguna untuk task ini adalah model klasifikasi yaitu Logistic Regression, K-Nearest Neighbor, Decision Tree, Random Forest dan XGBoost.

## The Training and Predict Scripts
1. Jalankan code berikut di terminal untuk menjalankan proses training dan testing
```python src/main.py```

2. Untuk menjalankan secara local, jalankan script berikut pada dua terminal yang berbeda
* Untuk service pada API
```python src/api.py```

* Untuk servic pada streamlit
```streamlit run streamlits.py```
