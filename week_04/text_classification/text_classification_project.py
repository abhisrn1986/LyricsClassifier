# %%
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
import time
import tqdm
import threading
import sys
import os
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder = os.path.realpath(dir_path+"/../"+"data/") + "/"

def extract_lyrics_from_url(url, songs, i):

    # time.sleep(10)
    print("Extrating from ", url)
    soup = BeautifulSoup(requests.get(url).text)
    lyrics = ""
    lyrics_tag = soup.find('pre', attrs={'id': 'lyric-body-text'})
    if lyrics_tag:
        for child in lyrics_tag.children:
            lyrics += child.text
    songs[i] = lyrics


def extract_songs(artist):
    artist_url = 'https://www.lyrics.com/artist/' + artist
    artist_html = requests.get(artist_url).text

    soup = BeautifulSoup(artist_html)
    songs = dict()
    square_bracket_pattern = ' [\[].*[\]]'
    link_constant = 'https://www.lyrics.com/'
    # for song in soup.find_all('strong'):
    for song in soup.find_all('td', attrs={'class': 'tal qx'}):
        a = song.find('strong').find('a')
        if a and not (re.findall(square_bracket_pattern, a.text)):
            songs[a.text.lower()] = link_constant + a.get('href')
    songs_df = pd.DataFrame(columns=["Title", "Link"])
    songs_df['Title'] = songs.keys()
    songs_df['Link'] = songs.values()

    # each thread extracts lyrics from each url
    all_lyrics = [None] * songs_df['Link'].shape[0]
    threads = []
    for index, url in enumerate(songs_df['Link'].values):
        t = threading.Thread(target=extract_lyrics_from_url,
                             args=[url, all_lyrics, index])
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()

    songs_df["Lyrics"] = all_lyrics

    return songs_df


def print_hypermaters_search_results(results):
    print('BEST MODEL PARAMETERS: {}\n'.format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    for mean, params in zip(means, results.cv_results_['params']):
        print('{}  for {}'.format(round(mean, 4), params))

# %%


parser = argparse.ArgumentParser(description="Run the script either for training a classifier"
                                 " to classify songs of two artists or to classify a given song"
                                 " by the two artists")

parser.add_argument('artists', type=str, nargs='+')

args = parser.parse_args()

artists = args.artists
artists_dfs = []

if len(artists) > 0:

    print(artists)

    for i, artist in enumerate(artists) :
        artist_filename = re.sub('[ -]{1}', "_", artist).lower() + '.csv'
        artist_filepath = data_folder + artist_filename
        if os.path.exists(artist_filepath) :
            artists_dfs.append(pd.read_csv(artist_filepath))
        else :
            artists_dfs.append(extract_songs(re.sub('[ _]{1}', "-", artist).lower()))
            artists_dfs[-1].to_csv(artist_filepath)

    # if REFRESH_LYRICS:
    #     imagine_dragons_df = extract_songs('Imagine-Dragons')
    #     imagine_dragons_df.to_csv("../data/imagine_dragons_songs.csv")

    #     linkin_park_df = extract_songs('Linkin-Park')
    #     linkin_park_df.to_csv("../data/linkin_park_songs.csv")
    # else:
    #     imagine_dragons_df = pd.read_csv("../data/imagine_dragons_songs.csv")
    #     linkin_park_df = pd.read_csv("../data/linkin_park_songs.csv")

    # %%
    # Create the lyrics data base
    df = pd.concat(artists_dfs)

    # convert the lyrics column type to string otherwise it is considered
    # as float
    df = df.assign(Lyrics=df["Lyrics"].astype(str))
    # remove all the \r from the lyrics
    X = df['Lyrics'].apply(lambda x: x.replace("\r", ""))

    # create targets
    # y_true = pd.Series([1] * imagine_dragons_df.shape[0] +
    #                    [0] * linkin_park_df.shape[0])

    y_list = []
    for i, artist_df in enumerate(artists_dfs):
        y_list = y_list + ([i] * artist_df.shape[0])
    y_true = pd.Series(y_list)



    # %%


    X_train, X_test, y_train, y_test = train_test_split(X, y_true, random_state=42)


    # %%

    vectorizer = CountVectorizer(
        lowercase=True, stop_words='english', token_pattern='[A-Za-z]+', ngram_range=(1, 1))

    nb_classifying_pipeline = Pipeline([
        ('vect', CountVectorizer(lowercase=True, stop_words='english',
        token_pattern='[A-Za-z]+', ngram_range=(1, 1))),
        ('model', MultinomialNB())])


    nb_classifying_pipeline.fit(X_train, y_train)
    y_pred = nb_classifying_pipeline.predict(X_test)
    accuracy_score(y_test, y_pred)


    # %%
    sgd_classifying_pipeline = Pipeline([
        ('vect', CountVectorizer(lowercase=True, stop_words='english',
        token_pattern='[A-Za-z]+', ngram_range=(1, 1))),
        ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

    # sgd_classifying_pipeline.fit(X_train, y_train)
    # y_pred = sgd_classifying_pipeline.predict(X_test)
    # accuracy_score(y_test, y_pred)

    # %%
    sgd_parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                    'clf__alpha': (1e-2, 1e-3)}

    sgd_grid_search_clf = GridSearchCV(
        sgd_classifying_pipeline, sgd_parameters, cv=5, n_jobs=-1, scoring='accuracy')

    sgd_grid_search_clf.fit(X_train, y_train)

    # print_hypermaters_search_results(sgd_grid_search_clf)

    # %%

    y_pred = sgd_grid_search_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
