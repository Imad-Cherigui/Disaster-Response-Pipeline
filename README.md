# Disaster-response-pipeline
### Description:
Building a web-app using machine learning to train and deploy a multioutput model designed to help classify messages into diffrent categories (fire, medical help...).
Such an application should prove helpful especially in time of crisis when first responders are flooded with messages and being able to correctly classify these messages may be critical giving the time sensitive nature of emergencies.


### How it works:
The project consists of a pipeline that:
- Injest the two data sets (messages and catagories), cleans and merges them (data/process_data.py) and then produces a database (data/data_base.db)
- Using the database created, it then trains and executes a multi output classifier and fine tunes it with GridSearch (models/process_data.py), the resulting model is stored an Pickle file (models/model.pkl) 
- the model is used by a web app to display the prediction of messages that the user input (app/run.py)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/data_base.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/data_base.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/


### Project structure
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # Python file to merge and clean the datasets
|- data_base.db   # database to save clean data to

- models
|- train_classifier.py # python file to build, train and export the model
|- model.pkl  # saved model 

- README.md
