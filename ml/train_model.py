import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
import sys
import logging
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from ml.model import inference, compute_slices

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load in the data
current_directory = os.getcwd()
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'census.csv'))
data = pd.read_csv(data_path, sep=", ", engine='python')

# Split the data into training and test sets
train, test = train_test_split(data, test_size=0.20, random_state=10,
                               stratify=data['salary'])
logging.info("Data split into training and test sets.")


# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(train,
                                             categorical_features=cat_features,
                                             label="salary", training=True)
logging.info("Training data processed.")


# Process the test data
X_test, y_test, encoder, lb = process_data(test,
                                           categorical_features=cat_features,
                                           label="salary", training=False,
                                           encoder=encoder, lb=lb)
logging.info("Test data processed.")

# Define the path to save model files
savepath = './model'
filenames = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Create the directory if it does not exist
os.makedirs(savepath, exist_ok=True)

# Check if the trained model exists on disk and load it
try:
    if all(os.path.isfile(os.path.join(savepath, fname))
           for fname in filenames):
        model = pickle.load(open(os.path.join(savepath, filenames[0]), 'rb'))
        encoder = pickle.load(open(os.path.join(savepath, filenames[1]), 'rb'))
        lb = pickle.load(open(os.path.join(savepath, filenames[2]), 'rb'))
        logging.info("Model and components loaded from disk.")
    else:
        # Train a new model and save it to disk
        model = train_model(X_train, y_train)
        pickle.dump(model, open(os.path.join(savepath, filenames[0]), 'wb'))
        pickle.dump(encoder, open(os.path.join(savepath, filenames[1]), 'wb'))
        pickle.dump(lb, open(os.path.join(savepath, filenames[2]), 'wb'))
        logging.info(f"Model and components saved to disk at: {savepath}")
except Exception as e:
    logging.error(f"Error during model loading or saving: {e}")
    sys.exit("Exiting due to error in model handling.")

# Compute model metrics
precision, recall, fbeta = compute_model_metrics(y_test,
                                                 inference(model, X_test))
logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(f"Precision: {precision:.3f},"
             f" Recall: {recall:.3f},"
             f" F-beta: {fbeta:.3f}")

# Define output directory for performance slices
output_dir = 'output_files'
os.makedirs(output_dir, exist_ok=True)

# Define the path to save the CSV file
slice_savepath = os.path.join(output_dir, 'performance_slices.csv')

# Compute and save performance slices
try:
    for feature in cat_features:
        performance_df = compute_slices(test, feature, y_test,
                                        inference(model, X_test))
        if os.path.exists(slice_savepath):
            performance_df.to_csv(slice_savepath, mode='a',
                                  header=False, index=False)
        else:
            performance_df.to_csv(slice_savepath, mode='w',
                                  header=True, index=False)
    logging.info(f"Performance slices saved to {slice_savepath}")
except Exception as e:
    logging.error(f"Error during performance slice computation: {e}")
    sys.exit("Exiting due to error in performance slice computation.")
