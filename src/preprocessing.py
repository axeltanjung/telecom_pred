import pandas as pd
import numpy as np
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [x_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [x_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [x_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set

def join_label_categori(set_data, config_data):
    # Check if label not found in set data
    if config_data["label"] in set_data.columns.to_list():
        # Create copy of set data
        set_data = set_data.copy()

        # Rename sedang to tidak sehat
        set_data.categori.replace(
            config_data["label_categories"][1],
            config_data["label_categories"][2], inplace = True
        )

        # Rename tidak tahu to tidak
        set_data.categori.replace(
            config_data["label_categories"][2],
            config_data["label_categories_new"][1], inplace = True
        )

        # Return renamed set data
        return set_data
    else:
        raise RuntimeError("Kolom label tidak terdeteksi pada set data yang diberikan!")

def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace -1 with NaN
    set_data.replace(
        -1, np.nan,
        inplace = True
    )

    # Return replaced set data
    return set_data


def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe_ContractRenewal = OneHotEncoder(sparse = False)

    # Fit ohe
    ohe_ContractRenewal.fit(np.array(data_tobe_fitted).reshape(-1, 1))

    # Save ohe object
    util.pickle_dump(
        ohe_ContractRenewal,
        ohe_path
    )

    # Return trained ohe
    return ohe_ContractRenewal

def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_ContractRenewal: OneHotEncoder) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Transform variable stasiun of set data, resulting array
    ContractRenewal_features = ohe_ContractRenewal.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    ContractRenewal_features = pd.DataFrame(
        ContractRenewal_features,
        columns = list(ohe_ContractRenewal.categories_[0])
    )

    # Set index by original set data index
    ContractRenewal_features.set_index(
        set_data.index,
        inplace = True
    )

    # Concatenate new features with original set data
    set_data = pd.concat(
        [ContractRenewal_features, set_data],
        axis = 1
    )

    # Drop ContractRenewal_features column
    set_data.drop(
        columns = "ContractRenewal_features",
        inplace = True
    )

    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data

def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 26)

    # Balancing set data
    x_rus, y_rus = rus.fit_resample(
        set_data.drop("Churn", axis = 1),
        set_data.Churn
    )

    # Concatenate balanced data
    set_data_rus = pd.concat(
        [x_rus, y_rus],
        axis = 1
    )

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 11)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(
        set_data.drop("Churn", axis = 1),
        set_data.Churn
    )

    # Concatenate balanced data
    set_data_ros = pd.concat(
        [x_ros, y_ros],
        axis = 1
    )

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(
        set_data.drop("Churn", axis = 1),
        set_data.Churn
    )

    # Concatenate balanced data
    set_data_sm = pd.concat(
        [x_sm, y_sm],
        axis = 1
    )

    # Return balanced data
    return set_data_sm

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(
        le_encoder,
        le_path
    )

    # Return trained le
    return le_encoder

def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()

    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_encoder_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    
    # Return transformed label data
    return label_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Join label categories
    train_set = join_label_categori(
        train_set,
        config_data
    )
    valid_set = join_label_categori(
        valid_set,
        config_data
    )
    test_set = join_label_categori(
        test_set,
        config_data
    )

    # 4. Converting -1 to NaN
    train_set = nan_detector(train_set)
    valid_set = nan_detector(valid_set)
    test_set = nan_detector(test_set)

    # 5. Handilng NaN ContractRenewal    
    # 5.1. Train set
    train_set.loc[train_set[(train_set.Churn == "Ya") & \
    (train_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Ya"]

    train_set.loc[train_set[(train_set.Churn == "Tidak") & \
    (train_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Tidak"]

    # 5.2. Validation set
    valid_set.loc[valid_set[(valid_set.Churn == "Ya") & \
    (valid_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Ya"]

    valid_set.loc[valid_set[(valid_set.Churn == "Tidak") & \
    (valid_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Tidak"]

    # 5.3. Test set
    test_set.loc[test_set[(test_set.Churn == "Ya") & \
    (test_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Ya"]

    test_set.loc[test_set[(test_set.Churn == "Tidak") & \
    (test_set.ContractRenewal.isnull() == True)].index, "ContractRenewal"] = \
    config_data["missing_value_ContractRenewal"]["Tidak"]

    # 6. Handling NaN DataPlan
    # 6.1. Train set
    train_set.loc[train_set[(train_set.Churn == "Ya") & \
    (train_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Ya"]

    train_set.loc[train_set[(train_set.Churn == "Tidak") & \
    (train_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Tidak"]

    # 6.2. Validation set
    valid_set.loc[valid_set[(valid_set.Churn == "Ya") & \
    (valid_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Ya"]

    valid_set.loc[valid_set[(valid_set.Churn == "Tidak") & \
    (valid_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Tidak"]

    # 6.3. Test set
    test_set.loc[test_set[(test_set.Churn == "Ya") & \
    (test_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Ya"]

    test_set.loc[test_set[(test_set.Churn == "Tidak") & \
    (test_set.DataPlan.isnull() == True)].index, "DataPlan"] = \
    config_data["missing_value_DataPlan"]["Tidak"]

    # 7. Handling Nan AccountWeeks, DataUsage, CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins
    impute_values = {
        "AccountWeeks" : config_data["missing_value_AccountWeeks"],
        "DataUsage" : config_data["missing_value_DataUsage"],
        "CustServCalls" : config_data["missing_value_CustServCalls"],
        "DayMins" : config_data["missing_value_DayMins"],
        "DayCalls" : config_data["missing_value_DayCalls"],
        "MonthlyCharge" : config_data["missing_value_MonthlyCharge"],
        "OverageFee" : config_data["missing_value_OverageFee"],
        "RoamMins" : config_data["missing_value_RoamMins"]
    }

    train_set.fillna(
        value = impute_values,
        inplace = True
    )
    valid_set.fillna(
        value = impute_values,
        inplace = True
    )
    test_set.fillna(
        value = impute_values,
        inplace = True
    )

    # 8. Fit ohe with predefined ContractRenewal data
    ohe_ContractRenewal = ohe_fit(
        config_data["range_ContractRenewal"],
        config_data["ohe_ContractRenewal_path"]
    )

    # 8. Fit ohe with predefined DataPlan data
    ohe_DataPlan = ohe_fit(
        config_data["range_DataPlan"],
        config_data["ohe_DataPlan_path"]
    )

    # 9. Transform ContractRenewal on train, valid, and test set
    train_set = ohe_transform(
        train_set,
        "ContractRenewal",
        ohe_ContractRenewal
    )

    valid_set = ohe_transform(
        valid_set,
        "ContractRenewal",
        ohe_ContractRenewal
    )

    test_set = ohe_transform(
        test_set,
        "ContractRenewal",
        ohe_ContractRenewal
    )

    # 9. Transform DataPlan on train, valid, and test set
    train_set = ohe_transform(
        train_set,
        "DataPlan",
        ohe_DataPlan
    )

    valid_set = ohe_transform(
        valid_set,
        "DataPlan",
        ohe_DataPlan
    )

    test_set = ohe_transform(
        test_set,
        "DataPlan",
        ohe_DataPlan
    )

    # 10. Undersampling dataset
    train_set_rus = rus_fit_resample(train_set)

    # 11. Oversampling dataset
    train_set_ros = ros_fit_resample(train_set)

    # 12. SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)

    # 13. Fit label encoder
    le_encoder = le_fit(
        config_data["label_categories_new"],
        config_data["le_encoder_path"]
    )

    # 14. Label encoding undersampling set
    train_set_rus.Churn = le_transform(
        train_set_rus.Churn, 
        config_data
    )

    # 15. Label encoding overrsampling set
    train_set_ros.Churn = le_transform(
        train_set_ros.Churn,
        config_data
    )

    # 16. Label encoding smote set
    train_set_sm.Churn = le_transform(
        train_set_sm.Churn,
        config_data
    )

    # 17. Label encoding validation set
    valid_set.Churn = le_transform(
        valid_set.Churn,
        config_data
    )

    # 18. Label encoding test set
    test_set.Churn = le_transform(
        test_set.Churn,
        config_data
    )

    # 19. Dumping dataset
    x_train = {
        "Undersampling" : train_set_rus.drop(columns = "Churn"),
        "Oversampling" : train_set_ros.drop(columns = "Churn"),
        "SMOTE" : train_set_sm.drop(columns = "Churn")
    }

    y_train = {
        "Undersampling" : train_set_rus.Churn,
        "Oversampling" : train_set_ros.Churn,
        "SMOTE" : train_set_sm.Churn
    }

    util.pickle_dump(
        x_train,
        "data/processed/x_train_feng.pkl"
    )
    util.pickle_dump(
        y_train,
        "data/processed/y_train_feng.pkl"
    )

    util.pickle_dump(
        valid_set.drop(columns = "categori"),
        "data/processed/x_valid_feng.pkl"
    )
    util.pickle_dump(
        valid_set.categori,
        "data/processed/y_valid_feng.pkl"
    )

    util.pickle_dump(
        test_set.drop(columns = "categori"),
        "data/processed/x_test_feng.pkl"
    )
    util.pickle_dump(
        test_set.categori,
        "data/processed/y_test_feng.pkl"
    )