# Machine Learning folder

## Description of data preprocessing

The data was categorized into integer columns for those related to the income, and object columns for those whose value was categorical; the dataset is made up mostly of categorical features therefore most features have values going from 1 to 10 and some string answers such as the marital status and profession.

The income columns were preprocessed by changing the values 999,998 and 999,999 to 0s since these values represent that the interviewee did not answer the question or they did not know their income. Those answers would be a categorical data inside a continuous feature thus they’re replaced by 0s.

All categorical features had their NaNs filled with ‘b’ which is the ENDIREH guideline value for blank answers. The continuous features had their NaNs filled with 0s. 

## Description of feature selection

The target questions for our research are those under the section P10_8 of the ENDIREH survey. These were made into a list to ease the process of filtering and selecting them.
Features that served as an identifier for the interviewee were removed from the dataset since these do not contribute to the machine learning model. Among these features are 'ID_VIV', 'ID_PER’,'UPM', 'VIV_SEL', 'HOGAR', 'N_REN'. 

Afterwards, redundant features were removed. We considered features to be redundant when they either had 1 obvious value like 'SEXO', 'COD_M15', 'CODIGO', 'COD_RES', 'COD_RES', or when their value was already contained in another question which is the case of 'COD_RES', 'CVE_ENT', 'CVE_MUN', 'GRA'.
Then, features which were related to the interview structure were removed, these include 'EST_DIS', 'UPM_DIS', 'ESTRATO', 'NOMBRE', 'REN_MUJ_EL', 'REN_INF_AD', 'N_REN_ESP', 'FAC_VIV', 'FAC_MUJ', 'PAREN'.

## Description of feature engineering

Feature P4_4 contains the profession of the interviewee; therefore, this information has a broad set of unique values which had to be bucketed according to the times each entry appeared. It was opted to grab the first word of each entry and group them according to this word. Then the answers with a length equal or lesser than 3, which groups most articles used in the job title were labeled as other. All categorical values which were left in the dataset were encoded using pandas get dummies.

The target questions from section P10_8 were saved in the y dataframe, while the relevant features were saved in the X dataframe.

Gower was used to create a distance matrix that accounts for both categorical and continous data and then DBSCAN was used using the precomputed distance matrix to create the clusters according to the answers given at each of the sections.

## Train test split

Using SKlearn train_test_split, the datasets were split by leaving 75% of the entries in the training dataset, and 25% for testing. The y dataset was stratified since the dataset is imbalanced.

## Model choice

The original choice for this task was a random forest classifier since it could output the feature importance of the dataset, however its performance for the minority class was not up to task. After further testing various models, the best performance came from the Imbalanced Learn EasyEnsembleClassifier and a Deep Neural Network. 

The neural network outputted the better results in regards of time of execution and has greater room for improvement. 

## Model training
Each of the target questions has its own X and y dataset. This is because the interviewees who answered one of the target questions may not have answered the following, thus the blank answers from each target question are removed from both X and y dataset. Negative answers were not removed since these are also relevant for the model.

### Further optimizations

The current options for optimizing performance are clustering the categorical information from the survey using an agglomerative clustering or finding a better alternative for this task. The model still needs an optimal number of clusters to work with.

Another option is adjusting the binary classification threshold using the ROC curve to find the optimal value that increases the True Positives and lowers the False Positive Rate.

### Accuracy score

The accuracy of the model oscilates between 69% for question P10_8_2 to 94% in question P10_8_10






