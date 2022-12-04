import preprocessing
import util as utils
import pandas as pd
import numpy as np

def test_join_label():
    # Arrange
    config = utils.load_config()

    mock_data = {
        "Churn" : [
            "Ya", "Tidak Tahu", "Tidak"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        "Churn" : [
            "Ya", "Tidak", "Tidak"]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.join_label_categori(mock_data, config)

    # Assert
    assert processed_data.equals(expected_data)

def test_nan_detector():
    # Arrange
    mock_data = {"missing_value_MonthlyCharge" : [23, -1, 50, 53, -1, 20]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"missing_value_MonthlyCharge" : [23, np.nan, 50, 53, np.nan, 20]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.nan_detector(mock_data)

    # Assert
    assert processed_data.equals(expected_data)

def test_ohe_transform():
    # Arrange
    config = utils.load_config()
    ohe_object = utils.pickle_load(config["ohe_ContractRenewal_path"])
    mock_data = {
        "ContractRenewal" : [
            "Renewal", "NotRenewal"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        "Renewal" : [1, 0], "NotRenewal" : [0, 1]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(float)

    # Act
    processed_data = preprocessing.ohe_transform(mock_data, "ContractRenewal", ohe_object)

    # Assert
    assert processed_data.equals(expected_data)

def test_le_transform():
    # Arrange
    config = utils.load_config()
    mock_data = {"Churn" : ["Tidak", "Ya", "Ya", "Ya", "Tidak", "Ya"]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"Churn" : [0, 1, 1, 1, 0, 1]}
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(int)

    # Act
    processed_data = preprocessing.le_transform(mock_data["Churn"], config)
    processed_data = pd.DataFrame({"Churn" : processed_data})

    # Assert
    assert processed_data.equals(expected_data)