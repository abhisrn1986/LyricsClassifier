import pandas as pd
import re
import os.path
import argparse
import logging

from text_models import get_sgd_trained_model
from web_scrapping import extract_songs

# Initialize the data folder to store the model and lyrics files
dir_path = os.path.dirname(os.path.realpath(__file__)) + "/"
data_folder = os.path.realpath(dir_path+"../"+"data/") + "/"

# specify the logging level
logging.basicConfig(encoding='utf-8', level=logging.INFO)


def ngram_type(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y,z")


if __name__ == "__main__":

    # Setup the command line arguements
    parser = argparse.ArgumentParser(description="Run the script either for training a classifier"
                                     " to classify songs of two or more artists or to classify a given song"
                                     " by the two or more artists")

    parser.add_argument('artists', type=str, nargs='+')
    parser.add_argument('--download', action='store_true',
                        help='Download songs to update songs csv files')
    parser.add_argument('--retrain', action='store_true',
                        help='Retrain the model from the data files')
    parser.add_argument('--ngrams', type=ngram_type, nargs='+',
                        default=[(1, 1), (1, 2)], help='ngrams list for grid search')

    predicting_args_grp = parser.add_argument_group(
        'Prediction functionality parameters', 'Parameters for predicting')
    predicting_args_grp.add_argument(
        '--predict', action='store_true', help='Flag for song artist prediction')
    predicting_args_grp.add_argument(
        '--song_files', type=str, nargs='+', help='Provide list of song files to predict the artists')

    sgd_args_grp = parser.add_argument_group(
        'Hyperparameters for sgd classifier', 'Hyperparameters for sgd classifier')
    sgd_args_grp.add_argument('--alphas', type=float, nargs='+',
                              default=[1e-2, 1e-3], help='List of alphas for sgd classifier')

    args = parser.parse_args()

    # make sure that the predict is set than song_files should be provided
    if (args.predict and not args.song_files) or (args.song_files and not args.predict):
        parser.error("Args --songs and --predict must occur together")

    artists = args.artists

    model_filename = re.sub('[ -]', '_', ''.join(artists)) + ".sav"
    model_filepath = data_folder + "models/" + model_filename


    # Check if there is a already a model saved for the artists combination
    # if yes get the model from the file if not generate the model
    if not os.path.exists(model_filepath) or args.retrain:

        artist_dfs = extract_songs(artists, data_folder, redownload=args.download)
        sgd_grid_search_clf = get_sgd_trained_model(
            model_filepath, artist_dfs, True, ngram_ranges=args.ngrams, alphas=args.alphas)

    else:
        sgd_grid_search_clf = get_sgd_trained_model(model_filepath)

    # predict the artist of the songs provided by the song files
    if args.predict:
        song_abs_filepaths = []
        for filepath in args.song_files:
            if os.path.exists(filepath):
                song_abs_filepaths.append(filepath)
            else:
                song_abs_filepaths.append(dir_path+filepath)

        # Create the test set from songs files provided
        file_songs = pd.Series([open(filepath, "r").read()
                               for filepath in song_abs_filepaths])
        X_test = file_songs.apply(lambda x: x.replace("\r", ""))
        # predict the songs
        y_pred = sgd_grid_search_clf.predict(X_test)

        for predict_artist_index in y_pred:
            print(artists[predict_artist_index])
