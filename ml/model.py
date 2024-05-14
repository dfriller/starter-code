from sklearn.metrics import (fbeta_score, precision_score, recall_score,
                             confusion_matrix)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import multiprocessing
import logging


logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    Use GridSearch for hyperparameter tuning and cross-validation

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    parameters = {
        'n_estimators': [10, 20, 30],
        'max_depth': [5, 10],
        'min_samples_split': [20, 50, 100],
        'learning_rate': [1.0],  # 0.1,0.5,
    }

    njobs = multiprocessing.cpu_count() - 1
    logging.info("Searching best hyperparameters on {} cores".format(njobs))

    clf = GridSearchCV(GradientBoostingClassifier(random_state=0),
                       param_grid=parameters,
                       cv=3,
                       n_jobs=njobs,
                       verbose=2,
                       )

    clf.fit(X_train, y_train)
    logging.info("********* Best parameters found ***********")
    logging.info("BEST PARAMS: {}".format(clf.best_params_))

    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_confusion_matrix(y, preds, labels=None):
    """
    Compute confusion matrix using the predictions and ground thruth provided
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    cm : confusion matrix for the provided prediction set
    """
    cm = confusion_matrix(y, preds)
    return cm


def compute_slices(df, feature, y, preds):
    """
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------
    df:
        test dataframe pre-processed with features as column used for slices
    feature:
        feature on which to perform the slices
    y : np.array
        corresponding known labels, binarized.
    preds : np.array
        Predicted labels, binarized

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """
    df['y'] = y
    df['preds'] = preds
    slices = df[feature].unique()
    df_performance = pd.DataFrame(columns=["feature", "n_samples",
                                           "precision", "recall", "fbeta"])

    for slice_value in slices:
        slice_data = df[df[feature] == slice_value]
        n_samples = len(slice_data)
        slice_data = slice_data.reset_index(drop=True)

        precision, recall, fbeta = compute_model_metrics(slice_data["y"],
                                                         slice_data["preds"])

        # Create a DataFrame from the dictionary and use concat
        new_row_df = pd.DataFrame([{
            'feature': slice_value,
            'n_samples': n_samples,
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        }])
        df_performance = pd.concat([df_performance, new_row_df],
                                   ignore_index=True)

    return df_performance
