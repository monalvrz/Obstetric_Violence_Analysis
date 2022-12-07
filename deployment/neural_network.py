# Initial imports.
import re
import gower
import sys

import pandas as pd
import sqlalchemy as sql

from sklearn.cluster import DBSCAN
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

    # Create a copy of the dataframe to avoid making changes in the original
    df = source_df.copy()

    print('Adjusting integer dtypes', file=sys.stderr)

    # First the income columns are saved in a list
    income_columns = ['P4_2', 'P4_5_AB', 'P4_7_AB', 'P4_9_1', 
                      'P4_9_2', 'P4_9_3', 'P4_9_4', 'P4_9_5', 
                      'P4_9_6', 'P4_9_7']

    # NA values are filled with 0s
    df[income_columns] = df[income_columns].fillna(0)

    # Dtypes are changed to float first in case the value is a string with a decimal, and then to int.
    df[income_columns] = df[income_columns].apply(lambda x: x.astype(float).astype(int)) 

    # Format the Income related columns since 999999 and 999998 are used to declare a non-specified income and thus can be used as 0
    df[(df[income_columns] >= 999998)][income_columns] = 0

    print('Adjusting string dtypes', file=sys.stderr)

    # Declare which features use text as their value (categorical features)
    string_columns = ['NOM_ENT', 'DOMINIO','P1_1','P1_4_1','P1_4_2','P1_4_3','P1_4_4','P1_4_5','P1_4_6','P1_4_7','P1_4_8',
                      'P1_4_9', 'P1_5', 'P1_6', 'P1_6', 'P1_8','P1_10_1','P1_10_2','P1_10_3','P1_10_4', 'P2_5','P2_6', 
                      'P2_8','P2_9','P2_10','P2_11','P2_12','P2_13','P2_14','P2_15', 'P2_16','P3_1','P3_2','P3_3','P3_4',
                      'P3_5','P3_6','P3_7', 'P3_8', 'P4AB_1', 'P4B_1','P4B_2','P4C_1','P4BC_3','P4BC_4','P4BC_5','P4_1',
                      'P4_2_1','P4_3', 'P4_4','P4_5_1_AB','P4_6_AB','P4_8_1','P4_8_2','P4_8_3','P4_8_4','P4_8_5','P4_8_6',
                      'P4_8_7', 'P4_10_2_1', 'P4_10_2_2', 'P4_10_2_3', 'P4_10_3_1', 'P4_10_3_2', 'P4_10_3_3','P4_11',
                      'P4_12_1','P4_12_2','P4_12_3','P4_12_4','P4_12_5','P4_12_6','P4_12_7', 'P4_13_1', 'P4_13_2', 'P4_13_3',
                      'P4_13_4', 'P4_13_5', 'P4_13_6', 'P4_13_7', 'P10_1_1','P10_1_2','P10_1_3','P10_1_4','P10_1_5','P10_1_6',
                      'P10_1_7','P10_1_8','P10_1_9','P10_5_01','P10_5_02','P10_5_03','P10_5_04','P10_5_05','P10_5_06','P10_5_07',
                      'P10_5_08','P10_5_09','P10_5_10','P10_5_11','P10_7']
    
    print('Cleaning int columns', file=sys.stderr)

    # Change the remaining columns to integer datatype
    df.loc[:,~df.columns.isin(string_columns)] = df.loc[:,~df.columns.isin(string_columns)].fillna(0)
    df.loc[:,~df.columns.isin(string_columns)] = df.loc[:,~df.columns.isin(string_columns)].astype(float).astype(int)

    print('Cleaning string columns', file=sys.stderr)

    # Change dtype of string columns to object
    df.loc[:,df.columns.isin(string_columns)] = df.loc[:,df.columns.isin(string_columns)].fillna('b')
    df.loc[:,df.columns.isin(string_columns)] = df.loc[:,df.columns.isin(string_columns)].astype(object)

    # Fill the remaining columns with b to represent they were left as blank
    df.fillna('b',inplace=True)

    # Create list of categorical columns
    categorical_features = df.dtypes[df.dtypes == 'object'].index.tolist()

    # Create target question list
    question_list = ['P10_8_1','P10_8_2','P10_8_3',
                      'P10_8_4','P10_8_5','P10_8_6',
                      'P10_8_7','P10_8_8','P10_8_9',
                      'P10_8_10','P10_8_11','P10_8_12',
                      'P10_8_13','P10_8_14','P10_8_15']

    print('Removing target columns from X dataset', file=sys.stderr)

    # Remove the target question from the list of categorical columns
    for question in question_list:
        if question in categorical_features:
            categorical_features.remove(question)

    print('Cleaning job column information', file=sys.stderr)

    # Split the answers in P4_4 and keep only the first word
    df['P4_4'] = df['P4_4'].str.split().str.get(0)

    # Bucket the P4_4 answers depending on their frequency 
    ## Create a dataframe to obtain the frequency of each answer for question P4_4
    answer_freq = pd.DataFrame({
                                'NAME':df['P4_4'].value_counts().index.tolist(),
                                'COUNT':list(df['P4_4'].value_counts())
                                })

    # Replace all answers that appeared less than 6 times in the dataset with Other
    for answer in list(answer_freq.loc[(answer_freq['COUNT']<6)]['NAME']):
      df['P4_4'] =df['P4_4'].replace(answer,"Other")
    
    # Replace all answers with a length equal or less than 3 in the dataset with Other
    for answer in list(answer_freq['NAME']):
      if len(answer)<=3:
        df['P4_4'] =df['P4_4'].replace(answer,"Other")

    # Set the categorical features dtype as string
    df[categorical_features].apply(lambda x: x.astype(str))

    print('Encoding dataframe', file=sys.stderr)

    # Enconde the categorical features
    encode_df = pd.get_dummies(df, columns=categorical_features, dtype=float)

    print('Storing final dataframe', file=sys.stderr)

    # Drop the rows where the target answers are blank
    df_X = encode_df.loc[encode_df[user_key] != 0].drop(columns=question_list)
    df_y = encode_df.loc[encode_df[user_key] != 0,[user_key]]

    # Create nested dictionary for the target question
    df_X_y_dict[user_key] = {}

    # Store the X and y datasets that will be used with the random forest model for the key question
    df_X_y_dict[user_key]['X'] = df_X
    df_X_y_dict[user_key]['y'] = df_y

    print(f'Received {len(df_X.columns)}', file=sys.stderr)
    return df_X_y_dict


##########################################################
# Function to predict the result using the trained model #
##########################################################

def Clustered_NN_Classifier(NN_model, NN_threshold, dict_X, Clustered_NN_Results = {}):

    # Create a copy of the X and y datasets to prevent modifications in the original dataset
    X = dict_X.copy()

    # Create list of columns that contain a survey answer except for the marital status question
    table_sections = ['P1','P2','P3','P4','P10']

    # Create dictionary to store the split column name groups
    section_features = {}

    print(f'Start clustering process with {len(X.columns)} features', file=sys.stderr)

    for table in table_sections:

      print(f'Creating cluster for table {table}', file=sys.stderr)

      #Group the encoded columns according to their starting code
      section_features[table] = [x for x in X.columns if not re.search("P3_8",x) if not re.search("P10_7",x) if re.search(f'{table}_',x)]

      # Create a dataframe that only has the survey answers columns
      survey_df = X[section_features[table]]

      # Create gower distance matrix
      distance_matrix = gower.gower_matrix(survey_df)

      # Configuring the parameters of the clustering algorithm
      dbscan_cluster = DBSCAN(eps=0.085, metric="precomputed", n_jobs=-1)

      # Fitting the clustering algorithm
      dbscan_cluster.fit(distance_matrix)

      # Add the cluster labels to the dataset
      X[f'{table}_Group'] = dbscan_cluster.labels_

      # Drop the columns from the original cluster
      X = X.drop(columns=section_features[table])

      # Enconde the clusters 
      X = pd.get_dummies(X, columns=[f'{table}_Group'])

    # These columns are the ones that were used when training the model, 
    # the clustering model might add a P_Group_-1 column if the input is an outlier and
    # would cause the model to crash so this is used as a safeguard
    training_model_columns = ['EDAD', 'NIV', 'P4AB_2', 'P4A_1', 'P4A_2', 
                              'P4BC_1', 'P4BC_2', 'NOM_ENT_AGUASCALIENTES', 
                              'NOM_ENT_BAJA CALIFORNIA', 'NOM_ENT_BAJA CALIFORNIA SUR', 
                              'NOM_ENT_CAMPECHE', 'NOM_ENT_CHIAPAS', 'NOM_ENT_CHIHUAHUA', 
                              'NOM_ENT_CIUDAD DE MÉXICO', 'NOM_ENT_COAHUILA DE ZARAGOZA', 
                              'NOM_ENT_COLIMA', 'NOM_ENT_DURANGO', 'NOM_ENT_GUANAJUATO', 
                              'NOM_ENT_GUERRERO', 'NOM_ENT_HIDALGO', 'NOM_ENT_JALISCO', 
                              'NOM_ENT_MICHOACÁN DE OCAMPO', 'NOM_ENT_MORELOS', 'NOM_ENT_MÉXICO', 
                              'NOM_ENT_NAYARIT', 'NOM_ENT_NUEVO LEÓN', 'NOM_ENT_OAXACA', 
                              'NOM_ENT_PUEBLA', 'NOM_ENT_QUERÉTARO', 'NOM_ENT_QUINTANA ROO', 
                              'NOM_ENT_SAN LUIS POTOSÍ', 'NOM_ENT_SINALOA', 'NOM_ENT_SONORA', 
                              'NOM_ENT_TABASCO', 'NOM_ENT_TAMAULIPAS', 'NOM_ENT_TLAXCALA', 
                              'NOM_ENT_VERACRUZ DE IGNACIO DE LA LLAVE', 'NOM_ENT_YUCATÁN', 
                              'NOM_ENT_ZACATECAS', 'DOMINIO_C', 'DOMINIO_R', 'DOMINIO_U', 'P3_8_A1', 
                              'P3_8_A2', 'P3_8_B1', 'P3_8_B2', 'P3_8_C1', 'P3_8_C2', 'P4AB_1_1.0', 
                              'P4AB_1_2.0', 'P4AB_1_3.0', 'P4AB_1_4.0', 'P4AB_1_b', 'P4B_1_1.0', 
                              'P4B_1_2.0', 'P4B_1_3.0', 'P4B_1_b', 'P4B_2_1.0', 'P4B_2_2.0', 
                              'P4B_2_3.0', 'P4B_2_4.0', 'P4B_2_5.0', 'P4B_2_6.0', 'P4B_2_7.0', 
                              'P4B_2_8.0', 'P4B_2_9.0', 'P4B_2_b', 'P4C_1_1.0', 'P4C_1_2.0', 
                              'P4C_1_8.0', 'P4C_1_b', 'P4BC_3_1.0', 'P4BC_3_2.0', 'P4BC_3_3.0', 
                              'P4BC_3_8.0', 'P4BC_3_b', 'P4BC_4_1.0', 'P4BC_4_2.0', 'P4BC_4_b', 
                              'P4BC_5_1.0', 'P4BC_5_2.0', 'P4BC_5_b', 'P10_7_1.0', 'P10_7_2.0', 
                              'P10_7_3.0', 'P10_7_4.0', 'P10_7_5.0', 'P10_7_6.0', 'P10_7_7.0', 
                              'P10_7_8.0', 'P10_7_9.0', 'P10_7_10.0', 'P10_7_b', 'P1_Group_-1', 
                              'P1_Group_0', 'P1_Group_1', 'P1_Group_2', 'P2_Group_-1', 'P2_Group_0', 
                              'P2_Group_1', 'P2_Group_2', 'P2_Group_3', 'P2_Group_4', 'P2_Group_5', 
                              'P3_Group_0', 'P3_Group_1', 'P3_Group_2', 'P3_Group_3', 'P3_Group_4', 
                              'P3_Group_5', 'P4_Group_0', 'P10_Group_-1', 'P10_Group_0']

    # Grab only the columns that the NN expects to receive
    X = X[training_model_columns]

    print(f'Scaling these inputs {X.iloc[-1].to_dict()}', file=sys.stderr)

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
