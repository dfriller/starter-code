# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Details
The goal of the model is to predict if an individual earns over $50K annually. It employs a GradientBoostingClassifier with hyperparameters optimized using GridSearchCV in scikit-learn 1.2.0. The chosen hyperparameters are:

learning_rate: 1.0
max_depth: 5
min_samples_split: 100
n_estimators: 10
The model is stored in a pickle file within the model directory. All training procedures and metrics are documented in "journal.log".


## Intended Use
This model is designed to estimate an individual's income level based on various attributes. It is intended primarily for educational, academic, or research purposes.

## Training Data
The training data, sourced from the UCI Machine Learning Repository's Census Income Dataset (Census Income Dataset), consists of 32,561 entries with 15 columns. These include a target label "salary" with two classes ('<=50K', '>50K'), 8 categorical, and 6 numerical features. The dataset exhibits a class imbalance with approximately 75% of the entries labeled '<=50K'. Initial data cleansing involved removing whitespace from data entries. Details on preprocessing steps are available in the "data_cleaning.ipynb" notebook. The data was split into an 80-20 ratio for training and testing, ensuring stratification based on the "salary" label. Categorical features were transformed using a One Hot Encoder, and the target label was processed with a label binarizer.


## Evaluation Data
The evaluation set comprises 20% of the dataset, with categorical features and the target label transformed using the same encoders as the training set.

## Metrics
The model's performance is assessed based on precision, recall, and F-beta scores, along with a confusion matrix. The results on the test set are:

Precision: 0.759
Recall: 0.643
F-beta: 0.696
Confusion Matrix: [[4625, 320], [560, 1008]]

## Ethical Considerations
It is critical to note that the dataset does not provide a comprehensive representation of current salary distributions and should not be used to infer specific population incomes.

## Caveats and Recommendations

The data derives from the 1994 Census database and may not accurately reflect the current demographic and economic conditions. It is advised to use this dataset for machine learning classification or similar studies, keeping in mind its limitations and historical context.
