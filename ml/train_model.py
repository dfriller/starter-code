# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pickle, os
import pandas as pd
from data import process_data
import os
import sys
from model import train_model, compute_model_metrics , inference, compute_slices
from model import compute_confusion_matrix
import logging

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
data = pd.read_csv("../data/census.csv", sep=", ", engine='python')

# Optional enhancement, use K-fold cross validation instead of a
# train-test split using stratify due to class imbalance
train, test = train_test_split( data,
                                test_size=0.20,
                                random_state=10,
                                stratify=data['salary']
                                )

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

X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Proces the test data with the process_data function.
# Set train flag = False - We use the encoding from the train set
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

savepath = '../model'
# Define the filenames for the stored model components
filenames = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# Check if the trained model exists on disk and load it
if os.path.isfile(os.path.join(savepath, filenames[0])):
    model = pickle.load(open(os.path.join(savepath, filenames[0]), 'rb'))
    encoder = pickle.load(open(os.path.join(savepath, filenames[1]), 'rb'))
    lb = pickle.load(open(os.path.join(savepath, filenames[2]), 'rb'))

# If the model does not exist, train a new model and save it to disk
else:
    # Train the model using the training data
    model = train_model(X_train, y_train)

    # Save the model and its components to disk in the specified directory
    pickle.dump(model, open(os.path.join(savepath, filenames[0]), 'wb'))
    pickle.dump(encoder, open(os.path.join(savepath, filenames[1]), 'wb'))
    pickle.dump(lb, open(os.path.join(savepath, filenames[2]), 'wb'))

    # Log the action of saving the model
    logging.info(f"Model and components saved to disk at: {savepath}")

precision, recall, fbeta = compute_model_metrics(y_test,inference(model,X_test))

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(
    f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")



output_dir = 'output_files'

# Create the directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the path to save the CSV file
slice_savepath = os.path.join(output_dir, 'performance_slices.csv')

for feature in cat_features:
    performance_df = compute_slices(test, feature, y_test, inference(model, X_test))

    # Check if the file already exists to avoid repeating headers
    if os.path.exists(slice_savepath):
        performance_df.to_csv(slice_savepath, mode='a', header=False, index=False)
    else:
        performance_df.to_csv(slice_savepath, mode='w', header=True, index=False)

sys.exit("Stopping the script after data processing.")








# Proces the test data with the process_data function.

# Train and save a model.
