import re
import os
import pandas as pd
import threading
from bs4 import BeautifulSoup
import requests
from sqlalchemy import desc
from tqdm import tqdm
import threading
import logging

class AtomicTqdm(tqdm):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._lock = threading.Lock()
        
    def update(self):
        with self._lock:
            super().update(n=1)
            

def extract_lyrics_from_url(url, songs, i, parser_for_soup):

    # if not hasattr(extract_lyrics_from_url.atomic_tqdm):
    #     extract_lyrics_from_url.atomic_tqdm = AtomicTqdm(total=urls_length)
    # else :
    #     extract_lyrics_from_url.atomic_tqdm = AtomicTqdm(total=urls_length)
    
    # print("Extrating from ", url)
    try : 
        soup = BeautifulSoup(requests.get(url).text, parser_for_soup)
    except:
        logging.warn("Exception occured while downloading from url: ", url, " skipping this song!")
    lyrics = ""
    lyrics_tag = soup.find('pre', attrs={'id': 'lyric-body-text'})
    if lyrics_tag:
        for child in lyrics_tag.children:
            lyrics += child.text
    songs[i] = lyrics
    # extract_lyrics_from_url.atomic_tqdm.update()


def extract_artist_songs(artist, parser_for_soup):
    artist_url = 'https://www.lyrics.com/artist/' + artist
    artist_html = requests.get(artist_url).text

    try : 
        soup = BeautifulSoup(artist_html, features=parser_for_soup)
    except :
        logging.warn("Exception occured while extracting song links for artist: ", artist, " skipping this artist!")
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

    # Each thread extracts lyrics from each url
    all_lyrics = [None] * songs_df['Link'].shape[0]
    
    # Create a thread for extracting each song of the current artist
    threads = []
    for index, url in enumerate(songs_df['Link'].values):
        t = threading.Thread(target=extract_lyrics_from_url,
                            args=[url, all_lyrics, index, parser_for_soup])
        t.start()
        threads.append(t)

    # Wait for all the songs to be downloaded.
    for thread in threads:
        thread.join()

    songs_df["Lyrics"] = all_lyrics
    
    # drop rows for any song for which lyrics were not extracted. 
    songs_df.dropna()

    return songs_df


def extract_songs(artists, songs_folder, redownload=False, parser_for_soup = "lxml"):
    # Extract the songs to data frames if there is a csv file else
    # web scrape the songs from the lyrics.com
    artists_dfs = []
    progress_bar = tqdm(total=len(artists), desc="Dowloading artist " + artists[0])
    n_artists = len(artists)
    for i, artist in enumerate(artists):

        artist_filename = re.sub('[ -]{1}', "_", artist).lower() + '.csv'
        artist_filepath = songs_folder + artist_filename
        if os.path.exists(artist_filepath) and not redownload:
            artists_dfs.append(pd.read_csv(artist_filepath))
        else:
            artists_dfs.append(extract_artist_songs(
                re.sub('[ _]{1}', "-", artist).lower(), parser_for_soup))
            artists_dfs[-1].to_csv(artist_filepath)
        
        if i < n_artists - 1:
            progress_bar.desc = "Dowloading artist " + artists[i+1]
        progress_bar.update(n=1)

    return artists_dfs
