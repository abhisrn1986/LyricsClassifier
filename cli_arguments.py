import argparse


def ngram_type(s):
    try:
        x, y = map(int, s.split(','))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Coordinates must be x,y,z")

# Setup the command line arguments parser


def get_cli_args():
    parser = argparse.ArgumentParser(description="Run the script either for training a classifier"
                                     " to classify songs of two or more artists or to classify a given song"
                                     " by the two or more artists")

    parser.add_argument('artists', type=str, nargs='+',
                        help='''provide a list of artists. Remember this should be a space seperated
                        list of artist names for example : \"Linkin Park\" \"Imagine Dragons\" \"ColdPlay\".''')
    parser.add_argument('--download', action='store_true',
                        help='''Download songs to update songs csv files locally.
                        When used it enforces that the songs are downloaded again
                        and csv files are updated. This is just a flag.''')
    parser.add_argument('--retrain', action='store_true',
                        help='''When this flag is enabled the text classifier model is retrianed from
                        the data files''')
    parser.add_argument('--ngrams', type=ngram_type, nargs='+',
                        default=[(1, 1), (1, 2)],
                        help='''ngrams list for hyperparameter grid search 
                        For example, --ngrams 1,1 1,3 this means [(1, 1), (1, 3)].''')

    predicting_args_grp = parser.add_argument_group(
        'Prediction functionality parameters', 'Parameters for predicting')
    predicting_args_grp.add_argument(
        '--predict', action='store_true',
        help='Flag for song artist prediction')
    predicting_args_grp.add_argument('--song_files', type=str, nargs='+',
                                     help='''Provide list of paths to song files
                                    to predict the artists and only the song file
                                    file names if song files are in ./data/songs
                                    folder''')

    sgd_args_grp = parser.add_argument_group(
        'Hyperparameters for sgd classifier', 'Hyperparameters for sgd classifier')
    sgd_args_grp.add_argument('--alphas', type=float, nargs='+',
                              default=[1e-2, 1e-3],
                              help='''List of alphas for sgd classifier
                              For example, --alphas 1e-2 1e-4''')

    args = parser.parse_args()

    # make sure that the predict is set than song_files should be provided
    if (args.predict and not args.song_files) or (args.song_files and not args.predict):
        parser.error("Args --songs and --predict must occur together")

    return args
