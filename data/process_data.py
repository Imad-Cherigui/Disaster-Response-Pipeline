import sys
import pandas as pd
import numpy as np
import sqlalchemy as sql

def load_data(messages_filepath, categories_filepath):
    """
    Load the files containing the messages and the categories and combine them in one dataframe using the "id" column as key
    
    Args:
        messages_filepath: The messages file
        categories_filepath: The categories file
        
    Returns:
        df: The combined dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.set_index("id").join(categories.set_index("id"))
    return df
    
def clean_data(df):
    """
    Cleans the dataframe:
    -Splits each category into its own column and naming it accordingly. 
    -converts values to binary (and corrects the values >1 to 1). 
    -Drops duplicates.
    
    Args:
        df: Merged messages and categories dataframe
        
    Returns:
        df: Clean dataframe
    """
    categories = df.categories.str.split(pat=";",expand=True)
    row = categories.iloc[0,:]
    category_colnames = [names[:-2] for names in row]
    categories.columns = category_colnames
    for column in categories.columns:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = categories[column].astype("int")
    df.drop(["categories"], axis=1, inplace = True) #delete the initial catagory column
    df = df.join(categories) #add the cleaned catagory columns to the dataframe
    improper_values = df.loc[df.related==2].related.index # filter the cells with a value of 2 instead of 1
    df.loc[improper_values,"related"]=1 # convert these values to 1
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """ 
    Stores the dataframe into a SQLite database,
    
    Args:
        df: the cleaned dataframe
        database_filename : destination path of SQLite database file.
     """
    engine = sql.create_engine(f"sqlite:///{database_filename}")
    df.to_sql('NLP_Table', engine, index=False)  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
