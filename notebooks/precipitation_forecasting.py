# %% [markdown]
# ## Importing modules and libraries

# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("../src")

from eda import calculate_statistics, plot_distributions
from missing_data import MissingDataAnalyzer, visualize_missing_patterns, handle_missing_values
from preprocess import load_data, scale_data, inverse_scaler
from ts_trainer import create_sequences, time_series_cross_validate, plot_time_series_splits, plot_loss_fuction
from visualization import plot_time_series, plot_predictions_vs_actual

import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError, Mean
import numpy as np

plt.style.use('seaborn-v0_8')

# %% [markdown]
# ## Load data

# %% [markdown]
# 

# %%
df = load_data('../data/raw/Precipitation.csv')

# %%
plot_time_series (
    df, 
    column='precipitation', 
    label='Precipitation',
    title='Precipitation Time Series',
    x_label='Year',
    y_label='Precipitation Amount (mm)')

# %% [markdown]
# ## Exploratory Data Analysis

# %%
calculate_statistics (data=df, column='precipitation')

# %%
plot_distributions (df, column='precipitation', plot_type = 'boxplot')

# %% [markdown]
# ### Missing Data Analysis

# %%
print(f"Missing Data: {df.missing.count_missing()}")
print(f"Present Data: {df.missing.count_complete()}")
df.missing.variable_summary()

# %%
shadow_matrix = (
    df
    .missing
    .bind_shadow_matrix(only_missing_cols=True)
)

shadow_matrix.groupby(['precipitation_NA'])['month_name'].describe().reset_index()

# %%
visualize_missing_patterns (df,'boxplot', x = 'month_name', value_col= 'precipitation')

# %%
visualize_missing_patterns (df,'histplot', x = 'year', value_col= 'precipitation')

# %% [markdown]
# ## Missing Data Handling

# %%
df_complete = handle_missing_values(df,value_col='precipitation')

# %%
plot_time_series (
    df_complete, 
    column='precipitation', 
    label='Precipitation',
    title='Precipitation Time Series',
    x_label='Year',
    y_label='Precipitation Amount (mm)')

# %%
calculate_statistics (df_complete, column='precipitation')

# %% [markdown]
# ## Scaling

# %%
data = df_complete ['precipitation'].values.copy() 
index = df_complete.reset_index().index.to_numpy()

test_size = 48
Train_size = len(data)-test_size
val_size = 48 

test_data =  data[Train_size:] 
Train_data = data[:Train_size] 
test_index =  index[Train_size:] 
Train_index = index[:Train_size] 

# %%
Train_data_scaled = scale_data (Train_data,test_data)[0]
test_data_scaled = scale_data (Train_data,test_data)[1]
scaler = scale_data (Train_data,test_data)[2]

# %%
plt.figure(figsize=(14,8))
plt.subplot (2,2,1)
plt.hist(Train_data, bins=12, color = 'royalblue', edgecolor = 'white')
plt.title('Training Set', fontsize=18)
plt.xlabel('Non Scaled Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.subplot (2,2,3)
plt.hist(Train_data_scaled, bins=12, color = '#F97306', edgecolor = 'white')
plt.xlabel('Scaled Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.subplot (2,2,2)
plt.hist(test_data, bins=12, color = 'royalblue', edgecolor = 'white')
plt.title('Test Set', fontsize=18)
plt.xlabel('Non Scaled Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.subplot (2,2,4)
plt.hist(test_data_scaled, bins=12, color = '#F97306', edgecolor = 'white')
plt.xlabel('Scaled Values', fontsize=14)
plt.ylabel('Frequency', fontsize=14)

plt.tight_layout()

# %% [markdown]
# ## Model Construction

# %%
input_length = 24 
output_length = 12

inputs = Input(shape=(input_length,1))
input_layer_out = LSTM (24,return_sequences=True)(inputs)
lstm_out_1 = LSTM(36)(input_layer_out)
outputs = Dense(12)(lstm_out_1)

model = Model(inputs=inputs, outputs=outputs)
model.compile (optimizer=Adam (learning_rate=0.0005), loss="mae",metrics=[RootMeanSquaredError])
model.summary()

# %% [markdown]
# ## Model trainig

# %%
model_training = time_series_cross_validate(
    model = model, 
    data = Train_data_scaled, 
    index = Train_index, 
    input_length = 24, 
    output_length = 12, 
    n_splits = 4, 
    val_size = val_size, 
    batch_size = 36, 
    epochs = 500, 
    verbose = 0)


# %%
plot_loss_fuction (model_training[0],True)

# %%
plot_loss_fuction (model_training[0],False)

# %%
plot_time_series_splits(tscv = model_training[1], time_series_data = data)

# %% [markdown]
# ## Model Prediction

# %%
x_test, y_test = create_sequences (test_data_scaled,input_length,output_length)
x_test = x_test.reshape(13,24)
y_test = y_test.reshape(13,12)

y_pred = model.predict(x_test,verbose=0)

predicted_values, actual_values = y_pred[len(y_pred)-1], y_test[len(y_pred)-1]

# Inverse scale
predicted_values, actual_values = inverse_scaler(scaler, predicted_values), inverse_scaler(scaler, actual_values)

# %% [markdown]
# ## Model evaluation

# %%
# Model evaluation
test_history = model.evaluate(x_test,y_test,verbose=0)

# Calculate prediction bias
bias = np.mean (np.array(y_pred).flatten() - (y_test.flatten()))

# Evaluation in original scale
evaluation = test_history
evaluation.append(bias)
print (f'Evaluation in range [-1,1]: \nMAE: {evaluation[0]}\n RMSE: {evaluation[1]}\n bias {evaluation[2]}')
evaluation = inverse_scaler(scaler, evaluation, metrics=True)
print (f'\nEvaluation in original range: \nMAE: {evaluation[0]}\n RMSE: {evaluation[1]}\n bias {evaluation[2]}')

# %%
plot_predictions_vs_actual(actual_values=actual_values, predicted_values=predicted_values,
                              title='Model Predictions vs Actual Values',
                              x_label='Time Period',
                              y_label='Value',
                              metrics={'MAE':evaluation[0], 'RMSE':evaluation[1], 'BIAS': evaluation[2]},
                              x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] )


