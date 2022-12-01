import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Prediksi Customer Churn pada Perusahaan Telecom")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create form of input
with st.form(key = "air_data_form"):
    # Create select box input
    ContractRenewal = st.selectbox(
        label = "1.\tApakah customer melakukan perpanjangan kontrak masa berlaku?",
        options = (
            "Ya",
            "Tidak",
        )
    )

    # Create select box input
    DataPlan = st.selectbox(
        label = "1.\tApakah customer menggunakan Data Plan untuk berkomunikasi?",
        options = (
            "Ya",
            "Tidak",
        )
    )

    # Create box for number input
    AccountWeeks = st.number_input(
        label = "2.\tTotal berapa minggu akun customer telah aktif?:",
        min_value = 0,
        max_value = 250,
        help = "Rentang nilai dari 0 hingga 250 minggu"
    )
    
    DataUsage = st.number_input(
        label = "3.\tBerapa total data usage perbulan yang digunakan customer (GB)?:",
        min_value = 0,
        max_value = 10,
        help = "Rentang nilai dari 0 hingga 10 GB"
    )

    CustServCalls = st.number_input(
        label = "4.\tBerapa total panggilan kepada customer service yang dilakukan?:",
        min_value = 0,
        max_value = 10,
        help = "Rentang nilai dari 0 hingga 10 kali"
    )

    DayMins = st.number_input(
        label = "5.\tBerapa menit rata-rata total penggunaan servis tiap bulan?:",
        min_value = 0,
        max_value = 375,
        help = "Rentang nilai dari 0 hingga 375 menit"
    )

    DayCalls = st.number_input(
        label = "6.\tBerapa jumlah rata-rata panggilan dalam sebulan:?",
        min_value = 0,
        max_value = 200,
        help = "Rentang nilai dari 0 hingga 200 menit"
    )

    MonthlyCharge = st.number_input(
        label = "7.\tBerapa rata-rata jumlah tagihan bulanan?:",
        min_value = 0,
        max_value = 150,
        help = "Rentang nilai dari 0 hingga 200 dollar"
    )

    OverageFee = st.number_input(
        label = "7.\tBerapa nilai tagihan terbesar dalam 12 bulan?:",
        min_value = 0,
        max_value = 150,
        help = "Rentang nilai dari 0 hingga 20 dollar"
    )
    
    RoamMins = st.number_input(
        label = "7.\tBerapa rata-rata waktu roaming?:",
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
            res = requests.post("http://api:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:
            if res["res"] != "Ya":
                st.warning("Prediksi Churn Customer: Tidak.")
            else:
                st.success("Prediksi Churn Customer: Ya.")