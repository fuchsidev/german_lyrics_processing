from collections import Counter

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import Secret

import re
import json
import os
from pathlib import Path

import lyricsgenius
from nltk import word_tokenize
from somajo import SoMaJo
from HanTa import HanoverTagger as ht
import pandas as pd


def get_lyrics(artist, file):
    import os.path
    if not os.path.exists(f"data/{artist}"):
        os.makedirs(f"data/{artist}")

    # Create and Configure Scraper
    genius = lyricsgenius.Genius(Secret.GENIUS_ACCESS_TOKEN)
    genius.verbose = True
    genius.remove_section_headers = False
    genius.skip_non_songs = True
    genius.excluded_terms = ["(Remix)", "(Snippet)"]

    # Create Artist Object, including all Songs
    result = genius.search_artist(artist, sort="title")
    result.save_lyrics(file, sanitize=False)


def lyrics_to_df(file):
    with open(file) as json_file:
        data = json.load(json_file)
    songs = data["songs"]
    song_data = []
    for song in songs:
        title = song["title"]
        album = song["album"]
        date = song["release_date"]
        lyrics = song["lyrics"].replace("EmbedShare URLCopyEmbedCopy", "").replace("'", "")
        # Remove e.g. [Hook]
        lyrics = re.sub("\[.*?\]", "", lyrics).replace("--", "")

        song_data.append({"artist": artist, "title": title, "album": album, "date": date, "lyrics": lyrics})
    df = pd.DataFrame(song_data)
    df.to_json(f"data/{artist}/filtered_data_{artist}.json")
    return df


def clean_data(df):
    # German Tagger to lemmatising (unifying words)
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    
    # German tokenizer to split sentences
    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    
    # Load Stopwords - words with no significant meaning to be removed from the lyrics.
    # Source: https://github.com/stopwords-iso/stopwords-de
    stop_words = ["nem", "--", "a", "ab", "aber", "ach", "acht", "achte", "achten", "achter", "achtes", "ag", "alle", "allein", "allem", "allen", "aller", "allerdings", "alles", "allgemeinen", "als", "also", "am", "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "au", "auch", "auf", "aus", "ausser", "ausserdem", "au??er", "au??erdem", "b", "bald", "bei", "beide", "beiden", "beim", "beispiel", "bekannt", "bereits", "besonders", "besser", "besten", "bin", "bis", "bisher", "bist", "c", "d", "d.h", "da", "dabei", "dadurch", "daf??r", "dagegen", "daher", "dahin", "dahinter", "damals", "damit", "danach", "daneben", "dank", "dann", "daran", "darauf", "daraus", "darf", "darfst", "darin", "darum", "darunter", "dar??ber", "das", "dasein", "daselbst", "dass", "dasselbe", "davon", "davor", "dazu", "dazwischen", "da??", "dein", "deine", "deinem", "deinen", "deiner", "deines", "dem", "dementsprechend", "demgegen??ber", "demgem??ss", "demgem????", "demselben", "demzufolge", "den", "denen", "denn", "denselben", "der", "deren", "derer", "derjenige", "derjenigen", "dermassen", "derma??en", "derselbe", "derselben", "des", "deshalb", "desselben", "dessen", "deswegen", "dich", "die", "diejenige", "diejenigen", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses", "dir", "doch", "dort", "drei", "drin", "dritte", "dritten", "dritter", "drittes", "du", "durch", "durchaus", "durfte", "durften", "d??rfen", "d??rft", "e", "eben", "ebenso", "ehrlich", "ei", "ei, ", "eigen", "eigene", "eigenen", "eigener", "eigenes", "ein", "einander", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "eins", "elf", "en", "ende", "endlich", "entweder", "er", "ernst", "erst", "erste", "ersten", "erster", "erstes", "es", "etwa", "etwas", "euch", "euer", "eure", "eurem", "euren", "eurer", "eures", "f", "folgende", "fr??her", "f??nf", "f??nfte", "f??nften", "f??nfter", "f??nftes", "f??r", "g", "gab", "ganz", "ganze", "ganzen", "ganzer", "ganzes", "gar", "gedurft", "gegen", "gegen??ber", "gehabt", "gehen", "geht", "gekannt", "gekonnt", "gemacht", "gemocht", "gemusst", "genug", "gerade", "gern", "gesagt", "geschweige", "gewesen", "gewollt", "geworden", "gibt", "ging", "gleich", "gott", "gross", "grosse", "grossen", "grosser", "grosses", "gro??", "gro??e", "gro??en", "gro??er", "gro??es", "gut", "gute", "guter", "gutes", "h", "hab", "habe", "haben", "habt", "hast", "hat", "hatte", "hatten", "hattest", "hattet", "heisst", "her", "heute", "hier", "hin", "hinter", "hoch", "h??tte", "h??tten", "i", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "immer", "in", "indem", "infolgedessen", "ins", "irgend", "ist", "j", "ja", "jahr", "jahre", "jahren", "je", "jede", "jedem", "jeden", "jeder", "jedermann", "jedermanns", "jedes", "jedoch", "jemand", "jemandem", "jemanden", "jene", "jenem", "jenen", "jener", "jenes", "jetzt", "k", "kam", "kann", "kannst", "kaum", "kein", "keine", "keinem", "keinen", "keiner", "keines", "kleine", "kleinen", "kleiner", "kleines", "kommen", "kommt", "konnte", "konnten", "kurz", "k??nnen", "k??nnt", "k??nnte", "l", "lang", "lange", "leicht", "leide", "lieber", "los", "m", "machen", "macht", "machte", "mag", "magst", "mahn", "mal", "man", "manche", "manchem", "manchen", "mancher", "manches", "mann", "mehr", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mensch", "menschen", "mich", "mir", "mit", "mittel", "mochte", "mochten", "morgen", "muss", "musst", "musste", "mussten", "mu??", "mu??t", "m??chte", "m??gen", "m??glich", "m??gt", "m??ssen", "m??sst", "m????t", "n", "na", "nach", "nachdem", "nahm", "nat??rlich", "neben", "nein", "neue", "neuen", "neun", "neunte", "neunten", "neunter", "neuntes", "nicht", "nichts", "nie", "niemand", "niemandem", "niemanden", "noch", "nun", "nur", "o", "ob", "oben", "oder", "offen", "oft", "ohne", "ordnung", "p", "q", "r", "recht", "rechte", "rechten", "rechter", "rechtes", "richtig", "rund", "s", "sa", "sache", "sagt", "sagte", "sah", "satt", "schlecht", "schluss", "schon", "sechs", "sechste", "sechsten", "sechster", "sechstes", "sehr", "sei", "seid", "seien", "sein", "seine", "seinem", "seinen", "seiner", "seines", "seit", "seitdem", "selbst", "sich", "sie", "sieben", "siebente", "siebenten", "siebenter", "siebentes", "sind", "so", "solang", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollen", "sollst", "sollt", "sollte", "sollten", "sondern", "sonst", "soweit", "sowie", "sp??ter", "startseite", "statt", "steht", "suche", "t", "tag", "tage", "tagen", "tat", "teil", "tel", "tritt", "trotzdem", "tun", "u", "uhr", "um", "und", "uns", "unse", "unsem", "unsen", "unser", "unsere", "unserer", "unses", "unter", "v", "vergangenen", "viel", "viele", "vielem", "vielen", "vielleicht", "vier", "vierte", "vierten", "vierter", "viertes", "vom", "von", "vor", "w", "wahr", "wann", "war", "waren", "warst", "wart", "warum", "was", "weg", "wegen", "weil", "weit", "weiter", "weitere", "weiteren", "weiteres", "welche", "welchem", "welchen", "welcher", "welches", "wem", "wen", "wenig", "wenige", "weniger", "weniges", "wenigstens", "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wessen", "wie", "wieder", "wieso", "will", "willst", "wir", "wird", "wirklich", "wirst", "wissen", "wo", "woher", "wohin", "wohl", "wollen", "wollt", "wollte", "wollten", "worden", "wurde", "wurden", "w??hrend", "w??hrenddem", "w??hrenddessen", "w??re", "w??rde", "w??rden", "x", "y", "z", "z.b", "zehn", "zehnte", "zehnten", "zehnter", "zehntes", "zeit", "zu", "zuerst", "zugleich", "zum", "zun??chst", "zur", "zur??ck", "zusammen", "zwanzig", "zwar", "zwei", "zweite", "zweiten", "zweiter", "zweites", "zwischen", "zw??lf", "??ber", "??berhaupt", "??brigens"]
    # New Columns which will be added to df
    lemming_lyrics, tokenized_lemming_lyrics, cleaned_token_lyrics = [], [], []
    
    # Iterate existing df
    for i, song in df.iterrows():
        lyrics = song["lyrics"]
        # Split text into individual lines, as presented on genius
        lines = lyrics.split("\n")
        # Tokenize lines to sentences, using the german tokenizer
        sentences = tokenizer.tokenize_text(lines)

        # Lemmatization of each word (coverting worrds to the base form, to unify e.g. "baum" and "b??ume" to "baum
        lemmatized_lyrics = ""
        for sentence_index, sentence in enumerate(sentences):
            for token_index, token in enumerate(sentence):
                tag = tagger.analyze(token.text)
                lemmatized_lyrics += f" {tag[0]}"
        # Tokenize lemming lyrics again
        token = word_tokenize(lemmatized_lyrics)
        # Remove german stopwords from token lyrics
        cleaned_token = [w for w in token if not w in stop_words]

        # Append cleaned lyrics to the respective lists
        lemming_lyrics.append(lemmatized_lyrics)
        tokenized_lemming_lyrics.append(token)
        cleaned_token_lyrics.append(cleaned_token)

    # Add cleaned lyric data to df and save to json
    df["Lemming Lyrics"] = lemming_lyrics
    df["Tokenized Lyrics"] = tokenized_lemming_lyrics
    df["Cleaned Tokenized Lyrics"] = cleaned_token_lyrics
    df.to_json(f"data/{artist}/cleaned_data_{artist}.json")
    return df


def process_data(df):
    """
    Things to analyze in Song lyrics:
    Rhyme Scheme
    Word frequency

    Sentiment Analysis (Positive or Negative Songs)
    Artist Development over time
    Compare to similar artist: Stolen Lyrics? Specific cliche words/lines?

    Count insults (clean song or not)
    How often are e.g. clothing brands or drugs mentioned by specific artists

    Total unique words
    Word cloud of songs or album in style of the logo (https://i.imgur.com/CsvEa5i.jpg)
    """
    all_tokens = []
    all_words = []
    for index, row in df.iterrows():
        all_words += row["Tokenized Lyrics"]
        all_tokens += row["Cleaned Tokenized Lyrics"]

    all_lyrics = " ".join(all_tokens)
    tokens = word_tokenize(all_lyrics)
    print("tokens", tokens)

    unique_words = len(set([x.lower() for x in all_words]))
    print("Unique Words:", unique_words)

    #logo_mask = np.array(Image.open(f"data/{artist}/logo.jpg"))
    wc = WordCloud(background_color="white", mask=logo_mask)
    wc.generate_from_frequencies(Counter(all_lyrics.split()))

    wc.to_file(f"data/{artist}/logo_wordcloud.png")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.figure()
    plt.show()


if __name__ == "__main__":
    artist_list = ["IDK & Offset", "Johann Wolfgang von Goethe", "Friedrich Schiller", "Heinrich Heine", "Shindy", "Credibil", "Die Fantastischen Vier"]

    for artist in artist_list:
        print(f"- Now handling Artist {artist}")
        # Ensure path for artist exists
        Path("/data/artist").mkdir(parents=True, exist_ok=True)
        raw_file = f"data/{artist}/raw_data_{artist}.json"
        if not os.path.isfile(raw_file):
            print("- Could not find raw data. Generating...")
            get_lyrics(artist, raw_file)
        print(f"- Raw data collected. Creating df...")
        filtered_df = lyrics_to_df(raw_file)
        print(f"- df created. Cleaning data...")
        cleaned_df = clean_data(filtered_df)
        print(f"- Data cleaned and saved.")
       # print(f"- Processing Data now")
       # process_data(cleaned_df)





