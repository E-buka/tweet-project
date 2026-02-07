# Tweet Sentiment Analysis

### Problem statement

Tweets have become a powerful way of communication and engaging with audience world wide. Approximately 500M tweets are made every single day ( Internet live stats). With each tweet comes a reaction and sentiment from various users. The aim of this project is to predict the sentiment a tweet will receive from the audience, either as positive or negative.  

#### Data
The dataset used for this project is a collection of 1.05M tweets classified as positive or negative. The data is available on Kaggle at  "https://www.kaggle.com/datasets/bhavikjikadara/tweets-dataset/data". Columns used in modelling are the text, date and target columns.

#### Project Structure

Pyspark was used for the entire project as in a realtime machine learning production pipeline though of a small scale. The data was cleaned and feature engineered using pyspark and a subsample of the cleaned data was visualized in Pandas and Matplotlib. The data modelling steps are saved as python scripts under a single package- tweet_analysis. A notebook version of the modelling is also available in the notebooks folder.  

##### Model
Data was split and saved as parquet for training and testing. The text data was transformed to spark TF_IDF while date column was engineered to generate date features. Model pipeline included both numeric and text features, but also with the option of using only text features. Notebook version shows modelling with text only and text+numeric features.  Cached TF-IDF was used to generate class weights from the training data before the final modelling using Pyspark ML Logistic Regression.  The pipeline model was saved.

##### Evaluation 
The pipeline model was reloaded and used to predict the test data and evaluated. Evaluation metrics used include AUC, F1-score, accuracy and confusion matrix. The AUC, F1 and accuracy are saved as json file and the confusion matrix saved as a parquet file in reports folder.

Finally, the model was deployed with FastAPI and Render as a demo for use with the link https://tweet-project-8395.onrender.com
 
The user is expected to enter a tweet on the API and get a sentiment prediction. 

### Deployment Notes

This service is containerized and production-ready.
When deployed on Render’s free tier (512MB RAM), the instance fails due to JVM memory requirements of PySpark.

Resolution:
Deploying the same container on ≥1GB RAM instances runs successfully.

This tradeoff is documented to highlight cost–infrastructure considerations when serving Spark models in real-time.


### How To Run locally (WSL/Ubuntu)
Run pip install -r requirements.txt to install necessary packages
1. Uncomment the file imports and __all__ var in the init.py file to run locally and comment out the deployment section.
2. Uncomment the section for running locally on config.py file and comment out the section for deployment.
3. Import and/or run the split.py file on terminal to clean the data generate numeric features and split the data into train and test data. 
4. A subsample of the data can be imported into eda.py for visualisation. 
5. Train the model by running the train.py file and save the pipeline model
6. Evaluate the test data by running main.py file
7. You can get predictions for sample tweets by running inference.py file to generate real time predictions. 
8. You can also get to host the model locally by running the api.py file with uvicorn or fastapi from terminal.  FastAPI and uvicorn are included in the requirements.txt file already if you installed by pip. 


##### Feature Improvements
For feature improvement, if the model is to be hosted with a paid server version, the model can be served using MLeap framework. There is also opportunity for batching and monitoring the model for model drift.  Retraining of the model with CI/CD will also be incorporated if this project expands in the future. 
