# Initial imports.
import re
import sys

import pandas as pd
import sqlalchemy as sql

from sklearn.preprocessing import StandardScaler

#######################################################
# Function to get the data stored in the RDS database #
#######################################################

def get_ML_dataset(user_entry):	

    print(user_entry)
    # Convert user inputs to series
    user_row = pd.DataFrame(user_entry,index=[0])
    print(user_row)

    # Drop the Target Question entry since its not relevant for the ML model
    user_row = user_row.drop(columns='Target_Question')

    print(f'These are the user inputs: \n {user_row.tail(1)}', file=sys.stderr)

    return user_row

############################################################################
# Function to split the data in X and y dataframes for the target question #
############################################################################

def DataFrame_X_y_split(user_key,source_df, df_X_y_dict = {}):
    
    # Create the chosen feature list
    feature_list = ['DOMINIO', 'EDAD', 
                    'P3_8', 'P10_2', 'P10_7']
    
    # Create a copy of the dataframe to avoid making changes in the original
    df = source_df.copy()
    
    # Grab only the target features
    df = df[feature_list]
    
    # Chose only the target features from the dataset and set dtype as string
    df[feature_list] = df[feature_list].fillna('b').astype(str)
    df['EDAD'] = df['EDAD'].astype(float).fillna(0).astype(int)

    # Enconde the categorical features
    encode_df = pd.get_dummies(df, dtype=float)
    print(encode_df)

    # The encoded labels that resulted from training the neural network
    NN_features = ['EDAD', 'DOMINIO_C', 'DOMINIO_R', 'DOMINIO_U',
                  'P3_8_A1', 'P3_8_A2', 'P3_8_B1', 'P3_8_B2',
                  'P3_8_C1', 'P3_8_C2', 'P10_2_1.0', 'P10_7_1.0',
                  'P10_7_10.0', 'P10_7_2.0', 'P10_7_3.0', 'P10_7_4.0',
                  'P10_7_5.0', 'P10_7_6.0', 'P10_7_7.0', 'P10_7_8.0',
                  'P10_7_9.0', 'P10_7_b']
    
    # Get the features which are missing from the user inputs 
    ## This is because the user input will only create 1 group, while the traning model had encoded groups for the whole dataset
    for column in encode_df.columns.to_list():
      NN_features.remove(column)  
    
    # Set the missing categorical features to 0.0
    for column in NN_features:
      encode_df[column] = 0.0

    # Create the dataset for the target question
    target = user_key

    # Drop the rows where the target answers are blank
    print(encode_df)
    df_X = encode_df

    # Create nested dictionary for the target question
    df_X_y_dict[target] = {}

    # Store the X and y datasets that will be used with the random forest model for the key question
    df_X_y_dict[target]['X'] = df_X

    print(df_X)

    return df_X_y_dict


##########################################################
# Function to predict the result using the trained model #
##########################################################

def NN_Classifier(NN_model, Saved_scaler, 
                  NN_threshold, dict_X, NN_Results = {}):

    # Create a copy of the X and y datasets to prevent modifications in the original dataset
    X = dict_X.copy()

    # Create a scaler instance
    X_scaler = Saved_scaler

    print(f'Scaling {len(X.iloc[-1])} inputs', file=sys.stderr)

    # Scale the X training data
    X_predict = X_scaler.transform(X.iloc[-1].values.reshape(1, -1))

    print(f'Predicting results for {X_predict}', file=sys.stderr)

    # Import the trained model
    nn = NN_model

    # Predict the results for the target question
    predictions = nn.predict(X_predict).ravel()

    # Convert predictions to 0 or 1 according to the optimal threshold
    threshold = NN_threshold

    # Label predictions using the threshold
    binary_predictions = (predictions >= threshold).astype(int)

    # Return answers to the predicted question
    if binary_predictions == 0:
      return f'Yes'
    else:
      return 'No'
