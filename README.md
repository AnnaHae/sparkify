# sparkify - data science capstone project 

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Project Descriptions](#descriptions)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code should run using Python versions 3* and pyspark version 3*. 

## Project Motivation<a name="motivation"></a>

The goal of this project is to build a machine learning model to predict customer churn. I used a dataset of Sparkify, a streaming service like Pandora or spotify.

Predicting customer churn is an essential task for companys to prevent users from logging off.
With a proper ML model it is possible to identify users who are likely to churn. These users then get special offers, discounts or incentives to make them stay. This can potentially save a business a lot in revenues.

The full dataset is 12GB, of which a tiny subset was analyzed. 
The model was build using Spark, Spark SQL, Spark dataframes and the machine learning APIs within Spark.


## File Descriptions <a name="files"></a>

There is one notebook available here to showcase work related to the above questions. Markdown cells were used to explain the individual steps. 

## Project Description <a name="descriptions"></a>

The following steps were performed to build the model:
   
1. Load and clean data
        The data was loaded using Sparks build in read.json method. Afterwards the nan and missing values were analyzed and removed for the userID and sessionID column.
 
 2. Explore data
        After step 1 the cleanded dataset was used to create a churn column that is used as a label for the machine learning model later.
        Using this column the dataset was further explored:
            - total churn ratio of unique users
            - gender churn ratio
            - subscription type churn ratio
            - time since registration
            - average number of items per session

3. Feature Engineering
        In this part of the projects features were build to train the model.
        Following features seemed promising:
            - days since registration
            - number of items per session
            - user level (paid/free subscription)
            - number of thumbs up/thumbs down
            - number of sessions
            - gender of user
            - total time in service
            - songs added to playlist
            - number of friends added
        The churn column was used as label for the model.
        All of those features and the label column were joined to the dataframe that was used for trianing and testing different ML models.
        
4. Build and Evaluate different ML models
        To train and evaluate different ML models a 'model_train_and_evaluate' function was used.
        This function performed several steps:
        1. Splitting dataset in training- and testset
        2. Building a pipeline that performs vectorization of the dataset using Vector Assembler, normalization of the dataset using Normalizer and classification of the dataset using different classifier (depending on input) 3. Fit pipeline on training set
        4. predict data on test set
        5. initialize MultiClassClassification Evaluater
        6. evaluate model on f1 score and accuracy
        The function was then used with different classifiers:
        - Logistic Regression
        - Linear Support Vector Machine
        - Decision Tree
        - Gradient Boosted Tree
        - Random Forest Classifier
        
    5. Tune Hyperparameters of ML Model
        The last step of the project was tuning the hyperparameters of the chosen model using ParamGridBuilder and CrossValidator. In this case xyxy was the model with the highest accuracy score and therefor tuning of hyperparameters was done with this model.Here the f1 score was used as metric to optimize, since the churned users are a fairly small subset in the dataset.
        
        
        
## Results<a name="results"></a>

The resulting ML model with the highest accuracy and the highest f1 score was Logistic Regression. 
The project shows that feature engineering is the esential part of building the model. Finding and implementing the right features is key to predicting customer churn with high accuracy.

For improvment of the model several steps are suggested:
- put additional effort in feature engineering (more time series analysis) and implement new features
- perform PCA on features to improve model run time
- perform analysis on a larger dataset (the analyzed churn rate was quite low)


The main findings of the projects can also be found at the post available [here](https://medium.com/@annatrumm/predicting-customer-churn-7469bb8af5b4).


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the data.
Otherwise, feel free to use the code here as you would like! 1

