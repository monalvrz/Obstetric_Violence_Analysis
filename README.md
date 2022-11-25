# Final_Project

## Presentation

For the final project we decided to address gender violence, specifically obstetric violence, i.e. any type of violence, both verbal and physical, that women experience during pregnancy. 

Our interest arose at the same time due to the information gathered by the ENDIREH (National Survey on the Dynamics of Household Relationships), conducted by INEGI (National Institute of Statistics and Geography) during 2021. As in the past edition of the survey (2016), the database included a group of additional questions regarding obstetric violence, which led us to think about the factors that could influence the violence experienced by women during pregnancy. 

The ENDIREH is the main reference on the situation of violence experienced by women in the country. The objective of this survey is to offer public information on the experiences of physical, economic, sexual, emotional and patrimonial violence that women 15 years of age and older have faced in the different areas of their lives, such as intimate partner, family, school, work and community. The survey makes it possible to estimate the main indicators on the prevalence and severity of violence, and at the same time to carry out a comparative exercise with those estimated in previous editions. 

The Survey is conducted every five years and is representative at the national and federative levels. Geographic coverage was divided into four categories: national, national urban, national rural and state. The sample size was 140,784 dwellings at the national level, with 4,371 and 4,426 dwellings as the minimum and maximum, respectively, per state. The sampling scheme was probabilistic, three-stage, stratified and clustered. And the survey period was from October 4 to November 30, 2021.  

From the survey questions and answers, 28 tables were created, in which the information collected is distributed. For this specific project we will work with the TVIV, TSDEM, TB_SEC_III, TB_SEC_IV and TB_SEC_X tables. 

Each of the tables contains the following information: 

-**TVIV**: contains the basic characteristics of the dwellings (services and goods available) as well as the number of persons usually residing there. 

-**TSDem**: contains the socio-demographic, economic and cultural characteristics of each of the persons residing in the dwelling. 

-**TB_SEC_III**: contains the information that identifies the women who are 15 years old and older who reside in the dwelling, and verifies the marital status of the respondent.

-**TB_SEC_IV**: contains basic information about the partner, husband, boyfriend or ex-partner of women who do not reside in the selected woman's household. It also provides information on the earned income of the woman and her partner, as well as other sources of income available to the woman. It delves into the assets and property owned by the persons residing in the dwelling, specifically the woman selected for the survey.

-**TB_SEC_X**: contains information on obstetric care received by women aged 15-49 who had a pregnancy during the last five years. 

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





## Communication protocols
The team's communication protocol is through Slack, where we have a specific group where we answer and resolve issues related to the project. In case of any emergency or urgent matter we have the phone numbers of each team member to respond as soon as possible. The group meets in the afternoons to advance and discuss the execution of the project.

The roles were distributed as follows:
- Square role: Mónica Alvarez
- Triangle role: Daniel Sañudo
- Circle role: Cristina Gonzalez 

Although everyone is in charge of a specific role, we all participate actively in the execution of each of the parts and sections that make up the project.
