from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import numpy as np
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
    # Convert data api to dataframe
    #datacolumns = data.columns
    #data = pd.DataFrame(data).T.set_index(datacolumns)
    # Convert dtype
    #data = pd.concat(
    #    [
    #        data[config_data["predictors"][0]],
    #        data[config_data["predictors"][1,2,3,5,7]].astype(int),
    #    ],
    #    axis = 1
    #)
    # Convert dtype
    data = pd.concat(
        [
            data[config_data["predictors"][0]].astype(str),
            data[config_data["predictors"][1:4]].astype(np.int32),
            data[config_data["predictors"][5:]].astype(np.float64)
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
    #y_pred = model_data.predict(data)

    # Inverse tranform
    y_pred = list(le_encoder.inverse_transform(y_pred))[0]

    #if y_pred[0] == 0:
    #    y_pred = "Tidak Churn"
    #else:
    #    y_pred = "Churn"
    #return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)