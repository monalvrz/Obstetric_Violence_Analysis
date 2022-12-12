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

    print('Connecting to AWS RDS', file=sys.stderr)

    # Create engine to connect to database
    engine = sql.create_engine(
      f'postgresql://postgres:bootcamp-project@obstetric-violence.clstnlifxcx7.us-west-2.rds.amazonaws.com:5432/ENDIREH_2021')

    # Get list of table names
    sql.inspect(engine).get_table_names()

    print('Reading Database', file=sys.stderr)

    # Read the obstetric_violence table and show the results
    RDS_df = pd.read_sql_table(
      'obstetric_violence', con=engine)

    # Creating a copy of the database to choose the features we will use to analyse
    df_copy = RDS_df.copy()

    print('Removing irrelevant columns', file=sys.stderr)

    # Remove columns that had data that wasn't usefull like ids, sampling information and table structure
    df_copy = df_copy.drop(columns=['ID_VIV', 'ID_PER' ,'UPM', 'VIV_SEL', 
                                    'HOGAR', 'N_REN', 'CVE_ENT', 'CVE_MUN', 
                                    'COD_RES', 'EST_DIS', 'UPM_DIS', 'ESTRATO', 
                                    'NOMBRE', 'SEXO', 'COD_M15', 'CODIGO', 'REN_MUJ_EL', 
                                    'REN_INF_AD', 'N_REN_ESP','T_INSTRUM', 'FAC_VIV', 
                                    'FAC_MUJ', 'PAREN', 'GRA', 'NOM_MUN', 'P4_4_CVE'])

    # Removing women that did not had a pregnancy on the last 5 years
    df_copy = df_copy[df_copy.P10_2 == 1.0].reset_index(drop=True)

    # Set dtype for question P4_10_3_3
    df_copy['P4_10_3_3'] = df_copy['P4_10_3_3'].astype(object)

    # Store the current dtypes to restablish them after getting the user inputs
    original_dtypes = df_copy.dtypes.to_dict()	

    # Convert user inputs to series
    user_row = pd.Series(user_entry)

    # Drop the Target Question entry since its not relevant for the ML model
    user_row = user_row.drop('Target_Question')

    print(f'These are the user input dtypes: {user_row}', file=sys.stderr)

    # Add the user inputs as a new row to the dataframe
    df_copy = pd.concat([df_copy,user_row.to_frame().T],ignore_index=True)

    # Restablish the dataframe dtypes
    df_copy= df_copy.astype(original_dtypes)

    print(f'These are the user inputs: {df_copy.tail(1)}', file=sys.stderr)

    for col in df_copy.columns:
      print(f'{col} {df_copy[col].dtype}') 
    return df_copy

############################################################################
# Function to split the data in X and y dataframes for the target question #
############################################################################

def DataFrame_X_y_split(user_key,source_df, df_X_y_dict = {}):

    # Create target question list
    question_list = ['P10_8_1','P10_8_2','P10_8_3',
                     'P10_8_4','P10_8_5','P10_8_6',
                     'P10_8_7','P10_8_8','P10_8_9',
                     'P10_8_10','P10_8_11']
    
    # Create the chosen feature list
    feature_list = ['DOMINIO', 'EDAD', 
                    'P3_8', 'P10_2', 'P10_7']
    
    # Create a copy of the dataframe to avoid making changes in the original
    df = source_df.copy()
    
    # Grab only the target features
    df = df[feature_list + question_list]
    
    # Chose only the target features from the dataset and set dtype as string
    df[feature_list] = df[feature_list].fillna('b').astype(str)
    df['EDAD'] = df['EDAD'].astype(float).fillna(0).astype(int)

    # Enconde the categorical features
    encode_df = pd.get_dummies(df, dtype=float)
    
    # Create the dataset for the target question
    target = user_key

    # Drop the rows where the target answers are blank
    df_X = encode_df.loc[(encode_df[target] == 1) | (encode_df[target] == 2)].drop(columns=question_list)
    df_y = encode_df.loc[(encode_df[target] == 1) | (encode_df[target] == 2),[target]]

    # Create nested dictionary for the target question
    df_X_y_dict[target] = {}

    # Store the X and y datasets that will be used with the random forest model for the key question
    df_X_y_dict[target]['X'] = df_X
    df_X_y_dict[target]['y'] = df_y

    return df_X_y_dict


##########################################################
# Function to predict the result using the trained model #
##########################################################

def NN_Classifier(NN_model, NN_threshold, dict_X, NN_Results = {}):

    # Create a copy of the X and y datasets to prevent modifications in the original dataset
    X = dict_X.copy()

    # Create a scaler instance
    scaler = StandardScaler()

    # Train the standard scaler using the X_train data
    X_scaler = scaler.fit(X.values)

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
      return 'Yes'
    else:
      return 'No'
