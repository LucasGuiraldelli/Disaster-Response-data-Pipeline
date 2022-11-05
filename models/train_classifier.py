import sys
import nltk
import pandas as pd
import re
import pickle

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def load_data(database_filepath):
    '''
        Load data from database as dataframe
        Input:
            database_filepath: File path of sql database
        Output:
            X: Message data (features)
            Y: Categories (target)
            category_names: Labels for 36 categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names


def tokenize(text):
    '''
        normalize, tokenizing, lemmatizing and removing stop words
        
        text: input the text
    '''
    
    text    = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # normalize text
    words   = word_tokenize(text) # tokenizing text
    words   = [w for w in words if w not in stopwords.words("english")] # removing stop words
    lemmedWords  = [WordNetLemmatizer().lemmatize(w) for w in words] # lemmatize
    
    
    return lemmedWords


def build_model():
    '''
        Build a ML pipeline using TF-IDF
        Input: None
        Output:
            pipeline results
    '''

    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        Evaluate model performance using test data
        Input: 
            model: Model to be evaluated
            X_test: Test data (features)
            Y_test: True lables for Test data
            category_names: Labels for 36 categories
        Output:
            Print accuracy and classfication report for each category
    '''

    y_prediction_test = model.predict(X_test)

    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_prediction_test[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(y_prediction_test[:, i], y_prediction_test[:,i])))


def save_model(model, model_filepath):
    '''
        Save model as a pickle file 
        Input: 
            model: Model to be saved
            model_filepath: path of the output pick file
        Output:
            A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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