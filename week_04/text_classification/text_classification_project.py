import numpy as np
import pandas as pd
import re
import time
import tqdm
import sys
import os
import argparse
import pickle

from web_scrapping import extract_songs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data_folder = os.path.realpath(dir_path+"../"+"data/") + "/"


def print_hypermaters_search_results(results):
    print('BEST MODEL PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    for mean, params in zip(means, results.cv_results_['params']):
        print('{}  for {}'.format(round(mean, 4), params))

if __name__ == "__main__":
    # Setup the command line arguements
    parser = argparse.ArgumentParser(description="Run the script either for training a classifier"
                                    " to classify songs of two artists or to classify a given song"
                                    " by the two artists")

    parser.add_argument('artists', type=str, nargs='+')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')

    predicting_args_grp = parser.add_argument_group('prediction functionality parameters', 'parameters for predicting')
    predicting_args_grp.add_argument('--predict', action='store_true', help='Predict the songs')
    predicting_args_grp.add_argument('--song_files', type=str, nargs='+', help='Provide list of song files to predict')

    args = parser.parse_args()

    if (args.predict and not args.song_files) or (args.song_files and not args.predict):
        parser.error("Args --songs and --predict must occur together")

    artists = args.artists

    # Check if there is a already a model saved for the artists combination
    model_filename = re.sub('[ -]','_',''.join(artists)) + ".sav"
    model_filepath = data_folder + "models/" + model_filename
    if not os.path.exists(model_filepath) or args.retrain :

        artist_dfs = extract_songs(artists, data_folder)
        # Create the lyrics data base
        df = pd.concat(artist_dfs)

        # convert the lyrics column type to string otherwise it is considered
        # as float
        df = df.assign(Lyrics=df["Lyrics"].astype(str))
        # remove all the \r from the lyrics
        X = df['Lyrics'].apply(lambda x: x.replace("\r", ""))

        # create targets
        y_list = []
        for i, artist_df in enumerate(artist_dfs):
            y_list = y_list + ([i] * artist_df.shape[0])
        y_true = pd.Series(y_list)

        # Use sgd classifier by default
        sgd_classifying_pipeline = Pipeline([
            ('vect', CountVectorizer(lowercase=True, stop_words='english',
            token_pattern='[A-Za-z]+', ngram_range=(1, 1))),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

        # Hyperparameter tunning
        sgd_parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                        'clf__alpha': (1e-2, 1e-3)}
        sgd_grid_search_clf = GridSearchCV(
            sgd_classifying_pipeline, sgd_parameters, cv=5, n_jobs=-1, scoring='accuracy')
        sgd_grid_search_clf.fit(X, y_true)
        # print_hypermaters_search_results(sgd_grid_search_clf)
        pickle.dump(sgd_grid_search_clf, open(model_filepath, 'wb'))
        print(sgd_grid_search_clf.score(X, y_true))

    else :
        sgd_grid_search_clf = pickle.load(open(model_filepath, "rb"))


    if args.predict :
        song_abs_filepaths = []
        for filepath in args.song_files:
            if os.path.exists(filepath):
                song_abs_filepaths.append(filepath)
            else:
                song_abs_filepaths.append(dir_path+filepath)
        
        # Create the test set from songs files provided
        file_songs = pd.Series([ open(filepath, "r").read() for filepath in song_abs_filepaths])
        X_test = file_songs.apply(lambda x: x.replace("\r", ""))
        # predict the songs
        y_pred = sgd_grid_search_clf.predict(X_test)

        for predict_artist_index in y_pred:
            print(artists[predict_artist_index])
