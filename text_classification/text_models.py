import logging
import pickle

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

LYRICS_COM_RANDOM_STATE = 42


def print_hypermaters_search_results(results):
    logging.info(f'\nBEST MODEL PARAMETERS: {results.best_params_}\n')
    means = results.cv_results_['mean_test_score']
    for mean, params in zip(means, results.cv_results_['params']):
        logging.info('{}  for {}'.format(round(mean, 4), params))


def get_train_test_data(artist_dfs):
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

    return train_test_split(X, y_true, random_state=LYRICS_COM_RANDOM_STATE, train_size=0.8)


def get_sgd_trained_model(model_filepath, artist_dfs=None, retrain=False, ngram_ranges=[(1, 1), (1, 2), (1, 3)], alphas=[1e-2, 1e-3, 1e-4], max_iters=[5, 10, 100, 1000]):
    # Create the lyrics data base
    if retrain:

        X_train, X_test, y_train, y_test = get_train_test_data(artist_dfs)

        # Use sgd classifier by default
        sgd_classifying_pipeline = Pipeline([
            ('vect', CountVectorizer(lowercase=True, stop_words='english',
                                     token_pattern='[A-Za-z]+', ngram_range=(1, 1))),
            ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=LYRICS_COM_RANDOM_STATE, max_iter=5, tol=None))])

        # Hyperparameter tunning
        sgd_parameters = {'vect__ngram_range': ngram_ranges,
                          'clf__alpha': alphas,
                          'clf__max_iter': max_iters}
        sgd_grid_search_clf = GridSearchCV(
            sgd_classifying_pipeline, sgd_parameters, cv=5, n_jobs=-1, scoring='accuracy')
        sgd_grid_search_clf.fit(X_train, y_train)
        print_hypermaters_search_results(sgd_grid_search_clf)
        with open(model_filepath, 'wb') as f:
            pickle.dump(sgd_grid_search_clf, f)
        logging.info(
            f"\nModel Accuracy: {sgd_grid_search_clf.score(X_test, y_test)}")
        return sgd_grid_search_clf

    else:
        with open(model_filepath, "rb") as f:
            return pickle.load(f)
