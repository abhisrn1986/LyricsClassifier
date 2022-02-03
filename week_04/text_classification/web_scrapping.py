import re
import os
import pandas as pd
import threading
import bs4 as BeautifulSoup
import requests

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


def extract_artist_songs(artist):
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

def extract_songs(artists, songs_folder) :
    # Extract the songs to data frames if there is a csv file else
    # web scrape the songs from the lyrics.com
    artists_dfs = []
    for i, artist in enumerate(artists) :
        artist_filename = re.sub('[ -]{1}', "_", artist).lower() + '.csv'
        artist_filepath = songs_folder + artist_filename
        if os.path.exists(artist_filepath) :
            artists_dfs.append(pd.read_csv(artist_filepath))
        else :
            artists_dfs.append(extract_artist_songs(re.sub('[ _]{1}', "-", artist).lower()))
            artists_dfs[-1].to_csv(artist_filepath)
    
    return artists_dfs