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

## Exploratory Analysis :mag_right:

For the exploratory analysis we used pandas to read the table we created by joining de five tables we chose and that conform the obstetric violence table. 

<img width="1008" alt="obstetric-table" src="https://user-images.githubusercontent.com/107893200/203886545-0105a0ba-ed1f-420d-9bb4-1ff9de9b5d19.png">

We then create the EDA dataframe that contains an overview of the information contained in the obstetric violence table.

<img width="733" alt="EDA-df" src="https://user-images.githubusercontent.com/107893200/203888189-fac05c91-e627-44b1-ba60-5805df466ffb.png">

We filtered the information to select only those women who answered that they had had a pregnancy in the last 5 years. Therefore, we made sure that women who answered *no* did not answer the questions related to obstetric care. 

## Machine Learning :gear:

### Model selection

The original choice for this task was a random forest classifier since it could output the feature importance of the dataset, however its performance for the minority class was not up to task. After further testing various models, the best performance came from the Imbalanced Learn EasyEnsembleClassifier and a Deep Neural Network.

The neural network outputted the better results in regards of time of execution and has greater room for improvement.

### Further optimizations
The current options for optimizing performance are clustering the categorical information from the survey using an agglomerative clustering or finding a better alternative for this task. The model still needs an optimal number of clusters to work with.

Another option is adjusting the binary classification threshold using the ROC curve to find the optimal value that increases the True Positives and lowers the False Positive Rate.

### Accuracy score
The accuracy of the model oscilates between 69% for question P10_8_2 to 94% in question P10_8_10

> For more information of the Machine Learning part of the project such as description od data preprocessing, description of feature selection, description of feature engineering, train test split, and model training you can redirect [here](https://github.com/monalvrz/Final_Project/tree/main/machine_learning).


## Dashboard :bar_chart:

If you want to access the dashboard of the final project you can review it [here](https://public.tableau.com/app/profile/cristina.gonzalez.zorrilla/viz/FinalProjectObstetricViolence/Story1?publish=yes).


https://user-images.githubusercontent.com/107893200/205732118-9e844415-d2ae-4114-9f89-b23daf1428f4.mov

### Tables

In addition to the dashboard graphs, we created five tables showing some of the percentages related to the women who answered the questions about obstetric care.

#### - Table 1.1

<img width="1289" alt="questions-table" src="https://user-images.githubusercontent.com/107893200/205209526-026dd818-38be-400f-9937-a2356897956a.png">

#### - Table 2.1

<img width="1026" alt="table-2 1" src="https://user-images.githubusercontent.com/107893200/205209559-19aa79ad-2f7d-4bb7-904c-85dfaa3643ae.png">

#### - Table 2.2

<img width="993" alt="table-2 2" src="https://user-images.githubusercontent.com/107893200/205209569-758df032-8fe4-4639-961b-38c1ae1bb62c.png">

#### - Table 2.3

<img width="993" alt="table-2 3" src="https://user-images.githubusercontent.com/107893200/205209585-da44b781-2f2e-46e9-a9de-1dcbc8963ef6.png">

#### - Table 2.4

<img width="545" alt="table-2 4" src="https://user-images.githubusercontent.com/107893200/205209589-0ccf2ad0-c381-4c8e-8d06-f2a63e5c4ce5.png">


