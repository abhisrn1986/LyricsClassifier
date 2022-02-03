from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import pandas as pd
import pickle
import os

def print_hypermaters_search_results(results):
    print('BEST MODEL PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    for mean, params in zip(means, results.cv_results_['params']):
        print('{}  for {}'.format(round(mean, 4), params))


def get_sgd_trained_model(model_filepath, artist_dfs=[], retrain=False):
    # Create the lyrics data base
    if retrain :
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
        print_hypermaters_search_results(sgd_grid_search_clf)
        pickle.dump(sgd_grid_search_clf, open(model_filepath, 'wb'))
        print(sgd_grid_search_clf.score(X, y_true))
        return sgd_grid_search_clf

    else :
        return pickle.load(open(model_filepath, "rb"))
    
