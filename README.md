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

```python src/data_pipeline.py```

```python src/preprocessing.py```

```python src/modeling.py```

2. Untuk menjalankan secara local, jalankan script berikut :

```sudo docker compose up```

## Flow Chart Permodelan Machine Learning
![End-to-end Maching Learning Pipeline](https://user-images.githubusercontent.com/87402782/205479152-99c8f3be-abb2-4544-b06d-fe4a276bca1a.png)

Berikut merupakan blok diagram yang terdiri:
* Block diagram persiapan (data_pipeline.py) - Hijau
* Block diagram preprocessing & features engineering (preprocessing.py) - Merah
* Block diagram modeling dan evaluasi (modeling.py) - Biru

## Format Message untuk melakukan prediksi via API dan reponnse dari API
### Format Message api.py
```
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()
ohe_ContractRenewal = util.pickle_load(config_data["ohe_ContractRenewal_path"])
le_encoder = util.pickle_load(config_data["le_encoder_path"])
model_data = util.pickle_load(config_data["production_model_path"])

class api_data(BaseModel):
    ID : int
    AccountWeeks : int   
    ContractRenewal : str
    DataPlan : int
    DataUsage : float
    CustServCalls : int
    DayMins : float
    DayCalls : int
    MonthlyCharge : float
    OverageFee : float
    RoamMins : float

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    # Convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][0]],
            data[config_data["predictors"][1:4]].astype(int),
            data[config_data["predictors"][4:]].astype(float)
        ],
        axis = 1
    )


    # Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    # Encoding ContractRenewal
    data = preprocessing.ohe_transform_ContractRenewal(data, ["ContractRenewal"], ohe_ContractRenewal)

    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0] 

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
    
```
### Format Message streamlit.py
```
import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('C:/Users/Axel/Desktop/Data Science/Telecom Prediction/assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Prediksi Customer Churn pada Perusahaan Telecom")
st.subheader("Isikan variable dibawah dah klik 'Predict':")

# Create form of input
with st.form(key = "telecom_data_form"):
        # Create box for number input
    ID = st.number_input(
        label = "1.Masukkan Nomor ID:",
        min_value = 0,
        max_value = 3333,
        help = "Rentang nilai dari 0 hingga 3333"
    )
    
    # Create select box input
    ContractRenewal = st.selectbox(
        label = "1.Apakah customer melakukan perpanjangan kontrak masa berlaku?",
        options = (
            "Renewal",
            "NotRenewal",
        )
    )

    # Create box for number input
    DataPlan = st.number_input(
        label = "2.Apakah customer menggunakan Data Plan untuk berkomunikasi? (Ya : 1 | Tidak : 0):",
        min_value = 0,
        max_value = 1,
        help = "Rentang nilai Ya : 1 | Tidak : 0"
    )

    # Create box for number input
    AccountWeeks = st.number_input(
        label = "3.Total berapa minggu akun customer telah aktif?:",
        min_value = 0,
        max_value = 250,
        help = "Rentang nilai dari 0 hingga 250 minggu"
    )
    
    DataUsage = st.number_input(
        label = "4.Berapa total data usage perbulan yang digunakan customer (GB)?:",
        min_value = 0,
        max_value = 10,
        help = "Rentang nilai dari 0 hingga 10 GB"
    )

    CustServCalls = st.number_input(
        label = "5.Berapa total panggilan kepada customer service yang dilakukan?:",
        min_value = 0,
        max_value = 10,
        help = "Rentang nilai dari 0 hingga 10 kali"
    )

    DayMins = st.number_input(
        label = "6.Berapa menit rata-rata total penggunaan servis tiap bulan?:",
        min_value = 0,
        max_value = 375,
        help = "Rentang nilai dari 0 hingga 375 menit"
    )

    DayCalls = st.number_input(
        label = "7.Berapa jumlah rata-rata panggilan dalam sebulan:?",
        min_value = 0,
        max_value = 200,
        help = "Rentang nilai dari 0 hingga 200 menit"
    )

    MonthlyCharge = st.number_input(
        label = "8.Berapa rata-rata jumlah tagihan bulanan?:",
        min_value = 0,
        max_value = 150,
        help = "Rentang nilai dari 0 hingga 200 dollar"
    )

    OverageFee = st.number_input(
        label = "9.Berapa nilai tagihan terbesar dalam 12 bulan?:",
        min_value = 0,
        max_value = 150,
        help = "Rentang nilai dari 0 hingga 20 dollar"
    )
    
    RoamMins = st.number_input(
        label = "10.Berapa rata-rata waktu roaming?:",
        min_value = 0,
        max_value = 25,
        help = "Rentang nilai dari 0 hingga 25 dollar"
    )


    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "ID": ID,
            "AccountWeeks":AccountWeeks,
            "ContractRenewal": ContractRenewal,
            "DataPlan": DataPlan,
            "DataUsage": DataUsage,
            "CustServCalls": CustServCalls,
            "DayMins": DayMins,
            "DayCalls": DayCalls,
            "MonthlyCharge": MonthlyCharge,
            "OverageFee": OverageFee,
            "RoamMins": RoamMins
        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8080/predict/", json = raw_data).json()

       
        
        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Ya":
                st.warning("Prediksi Churn Customer: Tidak.")
            else:
                st.success("Prediksi Churn Customer: Ya.")
 ```
 
 ## Cara menjalankan retraining model
 ### Retraining Model
 Jalankan code berikut di terminal untuk menjalankan proses training dan testing

```python src/data_pipeline.py```

```python src/preprocessing.py```

```python src/modeling.py```
 ### Running API
 
Untuk menjalankan secara local, jalankan script berikut:

```sudo docker compose up```
