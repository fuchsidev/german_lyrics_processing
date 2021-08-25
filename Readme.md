# Analyzing german song lyrics

This script scrapes the lyrics of given artists, cleans the data and (later) visualizes interesting aspects. 

The methods used to clean the data were selected for german song lyrics specifically using libraries that were developed for german language processing.


Current process:
1. Scrape Lyrics from Genius.com using [LyricsGenius](https://github.com/johnwmillr/LyricsGenius).  
2. Create pandas dataframe with the lyrics, song title etc.
3. Clean the data (Lemmatization, Stop words, tokenize)

## Requirements

```
pip install lyricsgenius
pip install pandas
pip install SoMaJo
pip install HanTa
pip install nltk
```

## License
[MIT](https://choosealicense.com/licenses/mit/)