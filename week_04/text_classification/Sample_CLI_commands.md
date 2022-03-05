## Samples for running the songs classifier



### Explaination of a cli command

In the below example, sgd model is trained using the songs from artists Linkin Park, Imagine Dragons and ColdPlay. --retrain is used here to train the model from scratch neglecting the a local model if it stored locally in a file. --predict is used when need to predict the songs to a respective artist and songs are provided by argument --song_files which is followed by list of song files (in folder ./data/songs). --ngrams and --alphas are list of hyperparameters used to train the model using grid search.

```shell
$ python text_classification_project.py "Linkin Park" "Imagine Dragons" "ColdPlay" --retrain --predict --song_files imagine_dragons_warriors.txt linkin_park_given_up.txt coldplay_magic.txt --ngrams 1,1 1,3 --alphas 1e-2 1e-4
```

Use this for printing the details on the arguments used.

```shell
$ python text_classification_project.py --help
```



### Other samples

```shell
$ python text_classification_project.py "Linkin Park" "Imagine Dragons" --predict --song_files imagine_dragons_warriors.txt linkin_park_given_up.txt --download --retrain
```

```shell
$ python text_classification_project.py "Linkin Park" "Imagine Dragons" --predict --song_files imagine_dragons_warriors.txt linkin_park_given_up.txt
```

```shell
$ python text_classification_project.py "Linkin Park" "Imagine Dragons" --predict --song_files imagine_dragons_warriors.txt linkin_park_given_up.txt --retrain
```

