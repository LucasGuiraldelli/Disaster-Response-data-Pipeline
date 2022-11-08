import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
        Input:
            messages_filepath: File path of messages data
            categories_filepath: File path of categories data
        Output:
            df: Merged dataset from messages and categories
    '''
    # Read message data
    messages = pd.read_csv(messages_filepath)
    # Read categories data
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
        Input:
            df: Merged dataset from messages and categories
        Output:
            df: Cleaned dataset
    '''
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').apply(lambda x:x[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    for column in categories:
        categories[column] = categories[column]\
                .astype(str).str.split('-').apply(lambda x:x[1]).astype('string').str.replace('2', '1')
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    df.drop_duplicates(subset='id', inplace=True)

    return df

def save_data(df, database_filename):
    '''
        Save df into sqlite db
        Input:
            df: cleaned dataset
            database_filename: database name
        Output: 
            SQL database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False) 


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