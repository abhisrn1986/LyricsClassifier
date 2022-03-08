# from xml.parsers.expat import model
import re
import os.path
from os import mkdir
import logging

import pandas as pd

from cli_arguments import get_cli_args
from text_models import get_sgd_trained_model
from web_scrapping import extract_songs

# Initialize the data folder to store the model and lyrics files
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data_folder = os.path.realpath(dir_path+"/"+"data/") + "/"

# specify the logging level
logging.basicConfig(encoding='utf-8', level=logging.INFO)

if __name__ == "__main__":

    # Process and get the cli arguments.
    args = get_cli_args()

    artists = args.artists

    model_filename = re.sub('[ -]', '_', ''.join(artists)) + ".sav"

    models_dir = os.path.join(data_folder, "models")
    if not os.path.exists(models_dir):
        mkdir(models_dir)

    model_filepath = models_dir + model_filename

    # Check if there is a already a model saved for the artists combination
    # if yes get the model from the file if not generate the model
    if not os.path.exists(model_filepath) or args.retrain:

        artist_dfs = extract_songs(
            artists, data_folder, redownload=args.download)
        sgd_grid_search_clf = get_sgd_trained_model(
            model_filepath, artist_dfs, True, ngram_ranges=args.ngrams, alphas=args.alphas)

    else:
        sgd_grid_search_clf = get_sgd_trained_model(model_filepath)

    # predict the artist of the songs provided by the song files
    default_songs_dir = os.path.join(data_folder, "songs/")
    if not os.path.exists(default_songs_dir):
        mkdir(default_songs_dir)

    if args.predict:
        song_abs_filepaths = []
        for filepath in args.song_files:
            if os.path.exists(filepath):
                song_abs_filepaths.append(filepath)
            else:
                song_abs_filepaths.append(default_songs_dir+filepath)

        # Create the test set from songs files provided
        file_songs = pd.Series([open(filepath, "r").read()
                            for filepath in song_abs_filepaths])
        X_test = file_songs.apply(lambda x: x.replace("\r", ""))
        # predict the songs
        y_pred = sgd_grid_search_clf.predict(X_test)

        for predict_artist_index in y_pred:
            print(artists[predict_artist_index])
