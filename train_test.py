import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from ml.model import inference, compute_model_metrics, compute_confusion_matrix
from ml.data import process_data
import pytest


@pytest.fixture(scope="module")
def data():
    """Fixture to load data from a CSV file."""
    datapath = "./data/census.csv"
    df = pd.read_csv(datapath)
    df.columns = df.columns.str.strip()
    return df


@pytest.fixture(scope="module")
def path():
    """Fixture to provide file path."""
    return "./data/census.csv"


@pytest.fixture(scope="module")
def features():
    """Fixture to return categorical features."""
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return cat_features


@pytest.fixture(scope="module")
def train_dataset(data, features):
    """Fixture to return a cleaned training dataset."""
    train, _ = train_test_split(
        data, test_size=0.20, random_state=10, stratify=data['salary']
    )
    X_train, y_train, _, _ = process_data(
        train, categorical_features=features, label="salary", training=True
    )
    return X_train, y_train


def test_import_data(path):
    """Test the presence and shape of the dataset file."""
    try:
        df = pd.read_csv(path)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err
    except AssertionError:
        logging.error("The file doesn't appear to have rows and columns")
        raise


def test_features(data, features):
    """Check that all categorical features are in dataset."""
    assert sorted(set(data.columns).intersection(features)) == sorted(features)


def test_is_model():
    """Check if the model file exists and can be loaded."""
    savepath = "./model/trained_model.pkl"
    try:
        with open(savepath, 'rb') as model_file:
            pickle.load(model_file)
    except FileNotFoundError:
        logging.error("Model file not found")
        raise
    except pickle.UnpicklingError:
        logging.error("The file does not appear to be a valid model file")
        raise


def test_is_fitted_model(train_dataset):
    """Verify that the model has been fitted."""
    X_train, _ = train_dataset
    model = pickle.load(open("./model/trained_model.pkl", 'rb'))
    try:
        model.predict(X_train)
    except NotFittedError as err:
        logging.error(f"Model is not fitted, error {err}")
        raise


def test_inference(train_dataset):
    """Check inference function."""
    X_train, _ = train_dataset
    model = pickle.load(open("./model/trained_model.pkl", 'rb'))
    preds = inference(model, X_train)
    assert preds is not None, "No predictions returned"
    assert preds.shape[0] == X_train.shape[0], "Prediction length mismatch"


def test_compute_model_metrics(train_dataset):
    """Check calculation of performance metrics."""
    X_train, y_train = train_dataset
    model = pickle.load(open("./model/trained_model.pkl", 'rb'))
    preds = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y_train, preds)


def test_compute_confusion_matrix(train_dataset):
    """Check calculation of the confusion matrix."""
    X_train, y_train = train_dataset
    model = pickle.load(open("./model/trained_model.pkl", 'rb'))
    preds = inference(model, X_train)
    cm = compute_confusion_matrix(y_train, preds)
    assert cm is not None, "Confusion matrix not generated"
    assert cm.shape == (2, 2), "Confusion matrix shape is incorrect"
