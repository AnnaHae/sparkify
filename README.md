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
    
    Metric: the metrics of choice are f1 score and accuracy. The accuracy is the proportion of correct predictions (churn customers and non churn customers). 
        
5. Tune Hyperparameters of ML Model
        The last step of the project was tuning the hyperparameters of the chosen model using ParamGridBuilder and CrossValidator. In this case xyxy was the model with the highest accuracy score and therefor tuning of hyperparameters was done with this model.Here the f1 score was used as metric to optimize, since the churned users are a fairly small subset in the dataset.
        
        
        
## Results<a name="results"></a>

In this project a spark environment is used to analyze a dataset. This gives us the opportunity to perform the analysis on a bigger dataset which needs more capacity and can usually not be performed on a single local machine.
Examine and predicting customer churn is an essential method for companies to prevent losing users.
This notebook shows that the main task is finding the right features to predict churn. It is crucial for implementing and improving the model.

For the evaluation i used the f1 score and accuracy. The f1 score is a measure of precision (sending the offer to the right person) and recall (missing users that we should have send an offer). The accuracy is a measure of how well we categorized the users in the two relevant classes ('churn' and 'non-churn').
Both help us adress the offers/incentives only to users that are likely to churn and help prevent sending offers to users, that aren't likely to churn. Sending out offers to these would mean wasting money because also without the offer they wouldn't want to churn.

For the features I´ve chosen i got the following model performances:
    
The classifier, that performed best on test set (regarding accuracy) was Logistic Regression.
The tuning of hyperparameters in that model only showed minimal improvements of f1 scores. 
* The logistic regression model has a accuracy of: 0.873, and F1 score of:0.814
* The linear support vector machine model has a accuracy of: 0.873, and F1 score of:0.814
* The decision tree model has a accuracy of: 0.817, and F1 score of:0.801
* The gradient boosted tree model has a accuracy of: 0.761, and F1 score of:0.769
* The random forest model has a accuracy of: 0.789, and F1 score of:0.783

The model performance of logistic regression and linear SVM had a higher accuracy and f1 score. This might be due to the simplicity of the model and that we only used very few features.
Since both models are limited when it comes to predicting complex models (i.e. when implementing new features) chosing GBT classifier, Decision Tree or Random Forest might lead to improved results.

As a final step hyperparameter tuning was performeds. The parameters `threshold` and `maxIter`of the Logistic Regression Model were tuned using paramGrid and crossVal and improving f1 score. The final parameters were: `threshold= 0.5` and `maxIter= 0` which led to the following model performance measured on the validation testset: f1-Score= 0.814  accuracy= 0.873.


The main findings of the projects can also be found at the post available [here](https://medium.com/@annatrumm/predicting-customer-churn-7469bb8af5b4).


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Udacity for the data.
Otherwise, feel free to use the code here as you would like! 1

