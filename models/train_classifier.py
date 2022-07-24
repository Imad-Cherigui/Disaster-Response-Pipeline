import sys
import pandas as pd
import sqlalchemy as sql
import re
import nltk
nltk.download(['punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
import pickle

def load_data(database_filepath):
    """ 
    Load data from database,
    Define feature and target variables X and Y
    
    Args:
        database_filepath: the database file path. 
        
    Returns: 
        X: Feature variables
        Y: Target variables
    """
    engine = sql.create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("NLP_Table", engine, index_col=None)
    X = df.message
    Y = df.loc[:,"related":]
    return X, Y

def tokenize(text):
    """
    Normalizes, lemmatizes, and tokenizes text.
    
     Args:
        text: A message. 
        
    Returns: 
        cleaned_tokens : Cleaned tokens of the message.
        
    """
    text = re.sub(r'[^\w\s]','',text) #cleans text
    tokens = word_tokenize(text) #tokenizes text
    lemmatizer = WordNetLemmatizer() #initiate the lemmatizer
    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]  
    return cleaned_tokens

def build_model():
    """ 
    Builds a pipeline with a multi-output classification models and then finetunes the model with GridSearch
    
    Returns:
       model_pipeline:  GridSearchCV model object. 
       
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])        
    parameters = {
            'clf__estimator__n_estimators' : [50,100,150,200] #the finetuning parameters
        }
    model_pipeline = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1) #Applies GridSearch to the Pipeline
    return model_pipeline

def evaluate_model(model, X_test, Y_test):
     """ 
     Evaluates the model for each category
    
    Args:
        model: the multi-output classification pipeline.
        X_test: Test features variables.
        Y_test: Test target variables.
        
    Prints:
        The classification reports for each catagory
    """
    y_pred = model.predict(X_test) # Makes the predictions
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index])) #Prints the results for each category

def save_model(model, model_filepath):
    """
    Saves the model in a Pickle file
    
    Args: 
        model: the multi-output classification pipeline.
        model_filepath: the path in which to save the pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
