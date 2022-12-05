# Analysis of ENDIREH Report on Obstetric Violence

## Presentation

For the final project we decided to address gender violence, specifically **obstetric violence**, i.e. any type of violence, both verbal and physical, that women experience during pregnancy. 

If you want to access the presentation of the final project you can review it [here](https://docs.google.com/presentation/d/1sfBlwcYBB2vFvdKs-UlIpPIlsR133COi-_0Lt6nKOMo/edit#slide=id.g1a888aca8e2_2_55).

### Why we selected this topic? :thought_balloon:

Our interest arose at the same time due to the information gathered by the **ENDIREH** (National Survey on the Dynamics of Household Relationships), conducted by INEGI (National Institute of Statistics and Geography) during **2021**. As in the past edition of the survey (2016), the database included a group of additional questions regarding obstetric care, which led us to think about the factors that could influence the violence experienced by women during pregnancy. 

### What is ENDIREH? :pushpin:

ENDIREH is the main reference on the situation of violence experienced by women in the country. The **objective** of this survey is to offer public information on the experiences of physical, economic, sexual, emotional and patrimonial violence that women 15 years of age and older have faced in the different areas of their lives, such as intimate partner, family, school, work and community. The survey makes it possible to estimate the main indicators on the prevalence and severity of violence, and at the same time to carry out a comparative exercise with those estimated in previous editions. 

The survey is conducted **every five years** and is representative at the national and federative levels. Geographic coverage was divided into four categories: national, national urban, national rural and state. The **sample size** was 140,784 dwellings at the national level, with 4,371 and 4,426 dwellings as the minimum and maximum, respectively, per state. The **sampling scheme** was probabilistic, three-stage, stratified and clustered. The survey period was from October 4 to November 30, 2021.  

### Description of the source data :card_index_dividers:

From the survey questions and answers, 28 tables were created, in which the information collected is distributed. For this specific project we will work with five tables: TVIV, TSDEM, TB_SEC_III, TB_SEC_IV and TB_SEC_X. 

Each of the tables contains the following information: 

-**TVIV**: contains the basic characteristics of the dwellings (services and goods available) as well as the number of persons usually residing there. 

-**TSDem**: contains the socio-demographic, economic and cultural characteristics of each of the persons residing in the dwelling. 

-**TB_SEC_III**: contains the information that identifies the women who are 15 years old and older who reside in the dwelling, and verifies the marital status of the respondent.

-**TB_SEC_IV**: contains basic information about the partner, husband, boyfriend or ex-partner of women who do not reside in the selected woman's household. It also provides information on the earned income of the woman and her partner, as well as other sources of income available to the woman. It delves into the assets and property owned by the persons residing in the dwelling, specifically the woman selected for the survey.

-**TB_SEC_X**: contains information on obstetric care received by women aged 15-49 who had a pregnancy during the last five years. 

### Questions we seek to answer :question:

From the process of selection and cleaning of the ENDIREH database, as well as the implementation of a machine learning model, we seek to answer the following questions: 

- What variables influence women to suffer obstetric violence?
- Women with partners are more likely to experience aggression from medical personnel?
- Is the region they live related to being exposed to obstetric violence?
- Is the socioeconomic level related to women who suffer obstetric violence?

## Exploratory Analysis

For the exploratory analysis we used pandas to read the table we created by joining de five tables we chose and that conform the obstetric violence table. 

~~~~
# Read the obstetric_violence table and show the results
df = pd.read_sql_table('obstetric_violence', con=engine)
df
~~~~

<img width="1008" alt="obstetric-table" src="https://user-images.githubusercontent.com/107893200/203886545-0105a0ba-ed1f-420d-9bb4-1ff9de9b5d19.png">

We then create the EDA dataframe that contains an overview of the information contained in the obstetric violence table.

~~~~
EDA_df = pd.DataFrame({
    'Dtype' : df.dtypes,
    'Number of Unique Values' : df.nunique(),
    'Number of Non-Empty Entries' : df.count(),
    'Number of Empty Entries' : df.isnull().sum(),
    'List of Values' : [df[col].value_counts().index.tolist() for col in df.columns],
})
~~~~

<img width="733" alt="EDA-df" src="https://user-images.githubusercontent.com/107893200/203888189-fac05c91-e627-44b1-ba60-5805df466ffb.png">

We filtered the information to select only those women who answered that they had had a pregnancy in the last 5 years. Therefore, we made sure that women who answered no did not answer the questions related to obstetric care. 

~~~~
pregnancy_not_in_last_5_years = df.loc[(df['P10_2'] == 2.0)]

target = ['P10_8_1',
'P10_8_2',
'P10_8_3',
'P10_8_4',
'P10_8_5',
'P10_8_6',
'P10_8_7',
'P10_8_8',
'P10_8_9',
'P10_8_10',
'P10_8_11',
'P10_8_12',
'P10_8_13',
'P10_8_14',
'P10_8_15']

for col in target:
    print(pregnancy_not_in_last_5_years[col].value_counts())
~~~~

## Machine Learning
The first thing we did to start with the selection of the most optimal machine learning model for the project was to decide which columns were or were not useful to find a relationship between obstetric violence and variables such as marital status, age, socioeconomic status. 

Therefore, we decided to discard the following columns: 
- 'ID_VIV', 'ID_PER' ,'UPM', 'VIV_SEL', 'HOGAR', 'N_REN', were discarded because their use is as an identifier of the respondent.
- 'N_REN' and 'REN_MUJ_EL', were discarded because they are used to reference a line item within the survey that is about another person.
- 'CVE_ENT', 'CVE_MUN', were discarded because we chose to preserve the name of the state and having the CVE of municipalities and states would be redundant.
- 'COD_RES', was discarded because the only value recorded in this column is 1.
- 'EST_DIS' and 'UPM_DIS', were discarded because they are variables referring to the sampling technique used.
- 'ESTRATO', was discarded because there is no clear information on what it means.
- 'NOMBRE', was discarded for two reasons, the person's name is not relevant to the model and no names were recorded for the respondents.
- 'SEXO', was discarded because it has only 1 value as a response.
- 'COD_M15', was discarded because it is used as an identifier for the respondents.
- 'CODIGO', was discarded because it is used as an identifier for the respondents.
- REN_INF_AD' and 'N_REN_ESP' were discarded because, like 'N_REN' and 'REN_MUJ', they are used to reference other data within the tables.
- T_INSTRUM' was discarded because it contains the same information as question P3_8.
- FAC_VIV' and 'FAC_MUJ' were discarded because they are weights calculated with the questions already being analyzed in the model.
- PAREN' was dropped because it is used to check the respondent's relationship with the head of household.
- GRA' was discarded because the VIN column contains redundant information about the degree of education of the person.
- NOM_MUN' was discarded because we chose to work with the states of the country instead of the municipality to avoid over-specifying the analysis.

First we reviewed the columns that had high values compared to the rest of the questions whose values varied between 0 and 100, and we found that the questions related to income reached a value of 999999. This value was used for people who did not answer the question, so these columns were filled in with 0 if they did not have a value, or if their value was 999999. After working with this data, the data type was changed to integer for these columns. 

The amount of features with string values such as the NOM_ENT (state name), NOM_MUN (city name), DOMINIO (domain), P3_8 (marital status) and P4_4 (current employment) is significantly less than those with numeric values, hence a list was created with these column names in order to filter them out when working with the numeric type data columns since the table had object and float dtypes for features that had integer data. These integer dtype features were preprocessed by fillling the NaNs with 0s, and changing their dtype to int. Afterwards, using the list of string value columns that was created previously, their NaNs were filled with 'b' since its the standard option for blank answers according tot he ENDIREH methodology.

After the NaNs were filled with the appropriate values, and the column dtypes were fixed. The next step was to bucket the P4_4 answers depending on their frequency, since around 90% of these answers only appeared once. The bucket threshold was picked to be 5, therefore the answers that appeared 4 or less times in the dataset were bucketed along the blank answers into the 'Other' group. This significantly reduced the dataset size from above 10,000 colums to 500.

After the bucket was created, the next step was to encode the categorical data. All columns whose dtype equaled object were encoded so that our classifier model could use this information.

Finally, in order to assess the different targets we have for this dataset, a nested dictionary was created; the outermost key belongs to the name of the target question, while its values correspond to the X dataset and the y target column. The X dataset was created by removing all target questions (all questions whose name starts with P10_8)  while the y values are those for the target column, The 0 values were removed from each X and y dataset using the target question as the guideline.

~~~~
def DataFrame_X_y_split(df,targets, df_X_y_dict = {}):
    # Format the Income related columns since 999999 is used to declare a non-specified income and thus can be used as 0
    income_columns = ['P4_2', 'P4_5_AB', 'P4_7_AB', 'P4_9_1', 'P4_9_2', 'P4_9_3', 'P4_9_4', 'P4_9_5', 'P4_9_6', 'P4_9_7']
    df[income_columns] = df[income_columns].fillna(0)
    df[income_columns].apply(lambda x: x.astype(int))    
    df[(df[income_columns] == 999999)][income_columns] = 0
    # Declare which features use text as their value (categorical features)
    string_columns = ['NOM_ENT','NOM_MUN', 'DOMINIO', 'P3_8','P4_4']
    # Change the remaining columns to integer datatype
    df.loc[:,~df.columns.isin(string_columns)] = df.loc[:,~df.columns.isin(string_columns)].fillna(0)
    df.loc[:,~df.columns.isin(string_columns)] = df.loc[:,~df.columns.isin(string_columns)].astype(int)
    # Fill the remaining columns with b to represent they were left as blank
    df.fillna('b',inplace=True)
    # Create list of categorical columns
    categorical_features = df.dtypes[df.dtypes == 'object'].index.tolist()
    # Remove the target question from the list of categorical columns
    for target in targets:
        if target in categorical_features:
            categorical_features.remove(target)
    print(f'List of categorical features: {categorical_features}')
    print(f'Number of unique entries in P4_4: {df["P4_4"].nunique()}')
    # Set the categorical features dtype as string
    df[categorical_features].apply(lambda x: x.astype(str))
    # Enconde the categorical features
    encode_df = pd.get_dummies(df, columns=categorical_features)
    # Create the dataset for each question
    for target in targets:
        # Drop the rows where the target answers are blank
        df_X = encode_df.loc[encode_df[target] != 0].drop(columns=targets)
        df_y = encode_df.loc[encode_df[target] != 0,[target]]
        # Create nested dictionary for the target question
        df_X_y_dict[target] = {}
        # Store the X and y datasets that will be used with the random forest model for the key question
        df_X_y_dict[target]['X'] = df_X
        df_X_y_dict[target]['y'] = df_y
    return df_X_y_dict
~~~~

A scaler was used in the X and y datasets to account for the difference in values between most answers and the income related answers.

The data was separated into train test using the train_test_split function of sklearn, the size was 75% for training and 25% for test. As these data are not balanced, we chose to use stratify in the y variable.

~~~~
[[ 176  171]
 [2563 1921]]
~~~~

We chose the Random Forest model of machine learning considering the following advantages and disadvantages: 

Advantages:

- It lessens decision tree overfitting and increases accuracy.
- It is adaptable to problems involving classification and regression.
- Both categorical and continuous values can be used with it.
- It automates filling in data's missing values.
- Data normalization is not necessary because a rule-based methodology is used.

Disadvantages:

- As it creates several trees to integrate their outputs, it uses a lot of resources and computational power.
- As it integrates numerous decision trees to decide the class, training takes a lot of time.
- It also suffers from interpretability issues and is unable to establish the relative importance of each variable because of the ensemble of decision trees.

## Dashboard 

If you want to access the dashboard of the final project you can review it [here](https://public.tableau.com/app/profile/cristina.gonzalez.zorrilla/viz/FinalProjectObstetricViolence/Dashboard1?publish=yes).

<img width="1289" alt="questions-table" src="https://user-images.githubusercontent.com/107893200/205209526-026dd818-38be-400f-9937-a2356897956a.png">

<img width="1026" alt="table-2 1" src="https://user-images.githubusercontent.com/107893200/205209559-19aa79ad-2f7d-4bb7-904c-85dfaa3643ae.png">

<img width="993" alt="table-2 2" src="https://user-images.githubusercontent.com/107893200/205209569-758df032-8fe4-4639-961b-38c1ae1bb62c.png">

<img width="993" alt="table-2 3" src="https://user-images.githubusercontent.com/107893200/205209585-da44b781-2f2e-46e9-a9de-1dcbc8963ef6.png">

<img width="545" alt="table-2 4" src="https://user-images.githubusercontent.com/107893200/205209589-0ccf2ad0-c381-4c8e-8d06-f2a63e5c4ce5.png">


