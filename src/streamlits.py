import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
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