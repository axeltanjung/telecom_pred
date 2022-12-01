from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("datetime").columns.to_list() == \
            params["datetime_columns"], "an error occurs in datetime column(s)."
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int32_columns"], "an error occurs in int32 column(s)."
    else:
        # In case checking data from api
        object_columns = params["object_columns"]
        del object_columns[1:]

        # Max column not used as predictor
        int_columns = params["int32_columns"]
        del int_columns[-1]

        # Check data types
        assert input_data.select_dtypes("int").columns.to_list() == params["int32_columns"], "an error occurs in int32 column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == params["float64_columns"], "an error occurs in float64 column(s)."
        assert input_data.select_dtypes("object").columns.to_list() == params["object_columns"], "an error occurs in object column(s)."

    assert set(input_data.ContractRenewal).issubset(set(params["range_ContractRenewal"])), "an error occurs in ContractRenewal range."
    assert set(input_data.DataPlan).issubset(set(params["range_DataPlan"])), "an error occurs in DataPlan range."
    assert set(input_data.Churn).issubset(set(params["range_Churn"])), "an error occurs in Churn range."
    assert input_data.ID.between(params["range_ID"][0], params["range_ID"][1]).sum() == len(input_data), "an error occurs in ID range."
    assert input_data.AccountWeeks.between(params["range_AccountWeeks"][0], params["range_AccountWeeks"][1]).sum() == len(input_data), "an error occurs in AccountWeeks range."
    assert input_data.DataUsage.between(params["range_DataUsage"][0], params["range_DataUsage"][1]).sum() == len(input_data), "an error occurs in DataUsage range."
    assert input_data.CustServCalls.between(params["range_CustServCalls"][0], params["range_CustServCalls"][1]).sum() == len(input_data), "an error occurs in CustServCalls range."
    assert input_data.DayMins.between(params["range_DayMins"][0], params["range_DayMins"][1]).sum() == len(input_data), "an error occurs in DayMins range."
    assert input_data.DayCalls.between(params["range_DayCalls"][0], params["range_DayCalls"][1]).sum() == len(input_data), "an error occurs in DayCalls range."
    assert input_data.MonthlyCharge.between(params["range_MonthlyCharge"][0], params["range_MonthlyCharge"][1]).sum() == len(input_data), "an error occurs in MonthlyCharge range."
    assert input_data.RoamMins.between(params["range_RoamMins"][0], params["range_RoamMins"][1]).sum() == len(input_data), "an error occurs in RoamMins range."

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )
    # 5. Handling Data Usage
    raw_dataset.DataUsage = raw_dataset.DataUsage.replace("----", -1).astype(float)

    raw_dataset.DataUsage.isna().sum()

    # 6. Handling CustServCalls
    raw_dataset.CustServCalls = raw_dataset.CustServCalls.replace("----", -1).astype(int)

    # 7. Handling CustServCalls
    raw_dataset.DayMins = raw_dataset.DayMins.replace("----", -1).astype(int)

    raw_dataset.DayMins.fillna(-1, inplace = True)
    
    # 8. Handling DayCalls
    raw_dataset.DayCalls = raw_dataset.DayCalls.replace("----", -1).astype(int)

    # 9. Handling MonthlyCharge
    raw_dataset.MonthlyCharge = raw_dataset.MonthlyCharge.replace("----", -1).astype(float)

    # 10. Handling OverageFee
    raw_dataset.OverageFee = raw_dataset.OverageFee.replace("----", -1).astype(float)

    raw_dataset.OverageFee.fillna(-1, inplace = True)

    # 11. Handling RoamMins
    raw_dataset.RoamMins = raw_dataset.RoamMins.replace("----", -1).astype(float)

    # 12. Handling variable Churn
    util.pickle_dump(
        raw_dataset,
        config_data["cleaned_raw_dataset_path"]
    )
    # 13. Check data definition
    check_data(raw_dataset, config_data)

    # 14. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.Churn.copy()

    # 15. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.3,
        random_state = 42,
        stratify = y
    )

    # 15. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.5,
        random_state = 42,
        stratify = y_test
    )

    # 16. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])