import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn import metrics

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
filename = 'california_housing_train.csv'
california_housing_dataframe = pd.read_csv(filename, sep=',')

# You can aswell reindex this data
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
    selected_features = california_housing_dataframe[[
        'latitude', 
        'longitude', 
        'housing_median_age', 
        'total_rooms', 
        'total_bedrooms', 
        'population', 
        'households', 
        'median_income']]

    processed_features = selected_features.copy()

    processed_features['rooms_per_person'] = processed_features['total_rooms']/processed_features['population']

    return processed_features

def preprocess_targets(california_housing_dataframe):
    output_targets = pd.DataFrame()
    output_targets['median_house_value'] = california_housing_dataframe['median_house_value']/1000.0
    
    return output_targets

# Split the training set into Training and Validation sets
training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
# print(training_examples.describe())

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))
# print(validation_examples.describe())


# Plot of the latitude and logitude data
plt.figure(figsize = (13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title('Validation Data Plot')

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])

plt.scatter(validation_examples['longitude'], validation_examples['latitude'], cmap = 'coolwarm', c = validation_targets['median_house_value']/validation_targets['median_house_value'].max())

ax = plt.subplot(1, 2, 2)
ax.set_title('Training Data Plot')

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])

plt.scatter(training_examples['longitude'], training_examples['latitude'], cmap = 'coolwarm', c = training_targets['median_house_value']/training_targets['median_house_value'].max())

plt.show()

# Train and Evaluate the model

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
    # Convert the panda data into a dictionary of np arrays
    features = {key:np.array(value) for key, value in dict(features).items()}

    # Construct the dataset ans configure the batching
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Check to see the shuffling
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of the dataset
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(my_feature) for my_feature in input_features])

#https://dl.google.com/mlcc/mledu-datasets/california_housing_test.csv
