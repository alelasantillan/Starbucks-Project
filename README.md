# Starbucks-Project
This is a partial requisit to fulfill the AWS Machine Learning Nano Degree by Udacity


## Historical
Stabucks is a well-known, relatively new retail company offering primarily coffee-based products that have taken over the world, and in particular the United States, by storm. Its success in changing Americans' coffee preferences is remarkable and so is the company's growth. Already in 2008 it began its digital transformation by offering loyalty cards and mobile apps, as well as promotional offers. These promotional offers have led to the generation of databases from which inferences can be drawn in order to offer promotional offers to its customers based on their responses to previous offers and thus find out which customers are more likely to respond to new offers.<br/>
From the above expanation one can deduce that for Starbucks and other similar companies point of view, the objective is to minimize customer churn.<br/>
In this respect, there is a lot of reasearch made, but I’ve concentrated in this paper in particular:<br/>
Predicting Customer Churn: Extreme Gradient Boosting with Temporal Data by Bryan Gregory (https://arxiv.org/pdf/1802.03396v1.pdf)<br/>
Because the customer churn research using machine learning is not new, so this paper is current, and it is closely related to the subject we are treating here. Besides, this paper won the WSDM Cup 2018 (*)<br/>

## Problem Statement
The problem to be solved is to determine whether each offer is accepted by the customer in question. In the past a customer may have received several offers or receives only one offer. The task is to classify (compute probability) whether or not the offer results in a change to the order (acceptance of the offer) based on the characteristics of the customer's data set.<br/>

## The Data
Data is provided by Starbucks and consists in three json files extracted from the Starbucks app. Please note that the data has been anonimized and 
reduced, since the whole unverse of offers sent to clients is composed by just 10 different choices. 

## Problem Resolution
We applied an XGBoost model for classifying the Starbucks customer's choices and the results show very accurate prediction.<br/>

## Instalation
This project runs only on AWS Sagemaker. <br/>
You will need an AWS account and open a Sagemaker Studio Session. <br/>
Upload the data as it is shown in the data folder.  <br/>
Upload the Jupyter Notebook and the train_model.py.  <br/>
The notebook should run fine. Let me know if you encounter any problem. <br/>
