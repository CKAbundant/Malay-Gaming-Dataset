"""Utilities functions required for generating synthetic
text-audio pair based on ChatGPT 3.5 corpus.
"""

import csv
import gc
import logging
import re
import string
import time
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import eng_to_ipa
import malaya
import pandas as pd
import parselmouth
import requests
from bs4 import BeautifulSoup
from omegaconf import DictConfig, OmegaConf
from pandarallel import pandarallel
from phonemizer.backend import EspeakBackend
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger

from .gaming_utils import (
    gen_audio,
    gen_filelist,
    gen_wav_file,
    pre_clean_text,
    remove_extra_whitespace,
    write_to_txt,
)
from .text.cleaners import (
    convert_to_ascii,
    expand_abbreviations,
    lowercase,
    malay_cleaners2,
)

# Initialize yaml path
YAML_PATH = Path(__file__).parents[2].joinpath("notebooks", "src_gaming", "gaming.yaml")
GPT_YAML_PATH = Path(__file__).parents[2].joinpath("notebooks", "src_gaming", "gaming_gpt.yaml")
polyglot_logger.setLevel("ERROR")

# Initialize global variables
TRANSFORMER = None
DICTIONARY = None
BACKEND_EN = None
BACKEND_MS = None


def initialize_pandarallel(progress_bar, num_proc: int):
    """Initialize pandarallel"""
    return pandarallel.initialize(progress_bar=progress_bar, nb_workers=num_proc)


def download_santai(cfg: DictConfig, abbrev: Optional[str] = None) -> List[str]:
    """Download transcripts from gamer santai and its sister sites' web page.

    Args:
        cfg (OmegaConf.DictConfig):
            OmegaConf dictionary containing url and file_paths to
            save transcripts from gamer santai.
        abbrev (Optional[str]):
            If provided, download transcript from specific web page abbreviation.

    Returns:
        (List[str]):
            List of transcript in sentence level.
    """

    mapping = {
        "senarai": (cfg.santai.url.senarai_path, cfg.santai.text.senarai_path),
        "dragonball": (cfg.santai.url.dragonball_path, cfg.santai.text.dragonball_path),
        "dragonball_1": (
            cfg.santai.url.dragonball_1_path,
            cfg.santai.text.dragonball_1_path,
        ),
        "polis": (cfg.santai.url.polis_path, cfg.santai.text.polis_path),
        "fakta": (cfg.santai.url.fakta_path, cfg.santai.text.fakta_path),
        "afghan": (cfg.santai.url.afghan_path, cfg.santai.text.afghan_path),
        "sega": (cfg.gamerwk.url.sega_path, cfg.gamerwk.text.sega_path),
    }

    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(mapping[abbrev][0]).content

    # Parse html content
    soup = BeautifulSoup(html_content, "lxml")

    # Extract transcript
    result = soup.find("div", class_="content-inner").text

    # Convert to list of transcript by sentence level
    paragraph = [sent.strip() for sent in result.split("\n") if sent]
    sentence = [text.strip() for sent in paragraph for text in sent.split(".") if text]

    # Save transcript to text file
    write_to_txt(sentence, mapping[abbrev][1])

    return sentence


def normalize_gpt(text: str) -> str:
    """Normalize gpt text based on custom mapping

    Args:
        text (str): Text string to be normalized.

    Returns:
        norm_text (str): Normalized gaming term.
    """

    # Load mapping from gaming.yaml
    cfg = OmegaConf.load(YAML_PATH)
    mapping = cfg.gpt_mapping

    # Replace slash, hyphen and brackets with white space
    # Replace apostrophe
    text = pre_clean_text(text)

    for term, replacement in mapping.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

    return remove_extra_whitespace(text)


def read_gpt(file_path: str) -> List[str]:
    """Read gpt.txt file and pre-process text strings.

    Args:
        file_path (str): String containing file path to gpt.txt

    Returns:
        clean_data (List[str]): List of formatted text strings.
    """

    data = []

    # Read gpt.txt
    with open(file_path, "r", encoding="utf-8") as f:
        # Remove new lines
        # data = [text.strip() for text in f.readlines() if text != "\n"]
        for text in f.readlines():
            # Remove "\n" character at end of text
            text = text.strip()

            # Consider only text string with text length of more than 1
            if text and len(text) > 1:
                # Split sentences into single sentence if any
                for sentence in split_into_sentences(text):
                    # Remove quote and brackets
                    sentence = re.sub(r"[\"\(\)]", "", sentence)

                    # Remove "Permain 1" and "Permain 2"
                    sentence = re.sub(r"Pemain \d: ", "", sentence)

                    # Append to data
                    data.append(sentence)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["text"])

    # Save cleaned text
    write_to_txt(df["text"].to_list(), file_path)

    return df


def split_into_sentences(text: str) -> List[str]:
    """Use a regex pattern to split sentences based on common
    sentence-ending punctuation.

    Args:
        text (str): text string.

    Returns:
        sentences(List[str])
    """

    sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    # Split the text into sentences.
    sentences = re.split(sentence_pattern, text)

    # Remove leading and trailing whitespaces from each sentence.
    sentences = [sentence.strip() for sentence in sentences]

    return sentences


def detect_language(text: str) -> Tuple[str, float]:
    """Function to detect language and return a tuple (language, confidence)"""

    try:
        detector = Detector(text)

        return (detector.language.code, detector.language.confidence)

    except Exception:
        return ("unknown", 0.0)


def extract_en_words(text: str) -> List[str]:
    """Extract english words from text string.

    Args:
        text (str): Text string containing english words.

    Returns:
        List[str]: List of english words.
    """

    return [word.lower() for word in text.split(" ") if detect_language(word)[0] == "en"]


def extract_gaming(text: str, combined_path: str) -> List[str]:
    """Extract gaming terms from text string.

    Args:
        text (str): Text string containing gaming terms.
        combined_path (str): Absolute path to `combined_gaming_terms.csv`.

    Returns:
        List[str]: List of gamimg terms.
    """

    gaming_terms = []

    # Read `combined_gaming_terms.csv`
    df_combined = pd.read_csv(combined_path)

    # Get list of combined gaming terms
    combined_terms = df_combined.loc[:, "term"].to_list()

    for term in combined_terms:
        # Regex pattern to identify standalone term, at start or at end of transcript
        pattern = pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"

        if re.search(pattern, text, flags=re.IGNORECASE):
            # Append term matches pattern
            gaming_terms.append(term)

    return gaming_terms


def gen_gaming_counter(data: pd.DataFrame, combined_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """Generate DataFrame containing frequency of gaming terms used in GPT corpus.

    Args:
        data (pd.Dataframe):
            DataFrame containing transcripts and `gaming` column which contains
            list of all gaming terms found.
        combined_path (str):
            Absolute path to `combined_gaming_terms.csv`.

    Returns:
        df_gaming_counter (pd.DataFrame):
            DataFrame containing frequency of gaming terms used.
        unused_gaming (List[str]):
            List of gaming terms that were not used in GPT corpus.
    """

    # Initialize counter object to count gaming terms
    gaming_counter = Counter()

    # Update gaming_counter
    for terms in data["gaming"]:
        gaming_counter.update(terms)

    # Load `combined_gaming_terms.csv` as DataFrame
    combined = pd.read_csv(combined_path)

    # Get list of unique gaming terms and actual gaming terms used
    combined_gaming = combined["term"].to_list()
    used_gaming = list(gaming_counter.keys())

    # Get list of unused gaming terms
    unused_gaming = list(set(combined_gaming) - set(used_gaming))

    # Display statistics
    total_gaming = len(combined_gaming)
    num_used = len(used_gaming)
    num_unused = len(unused_gaming)

    print(f"{'Total number of combined gaming terms':<40} : {total_gaming}")
    print(f"{'Number of gaming terms used':<40} : {num_used}")
    print(f"{'Number of gaming terms unused':<40} : {num_unused}")

    # Convert gaming_counter to DataFrame
    df_gaming_counter = pd.DataFrame.from_dict(gaming_counter, orient="index")

    # Rename columns and sort by decreasing frequency count
    df_gaming_counter.columns = ["frequency"]
    df_gaming_counter = df_gaming_counter.sort_values(by="frequency", ascending=False)

    return df_gaming_counter, unused_gaming


def gen_en_counter(data: pd.DataFrame) -> pd.DataFrame:
    """Generate DataFrame containing frequency of gaming terms used in GPT corpus.

    Args:
        data (pd.Dataframe):
            DataFrame containing transcripts and `en_words` column which contains
            list of all English words detected by polyglot.

    Returns:
        df_en_counter (pd.DataFrame):
            DataFrame containing frequency of gaming terms used.
    """

    # Initialize counter object to count English words detected
    en_counter = Counter()

    # Update gaming_counter
    for words in data["en_words"]:
        if isinstance(words, str):
            # Pandas saved List as a str
            en_counter.update(literal_eval(words.lower()))

        elif isinstance(words, list):
            # Lower case all words
            words = [word.lower() for word in words]
            en_counter.update(words)

    print(f"Total number of English words detected by polyglot : {len(en_counter)}")

    # Convert gaming_counter to DataFrame
    df_en_counter = pd.DataFrame.from_dict(en_counter, orient="index")

    # Rename columns and sort by decreasing frequency count
    df_en_counter.columns = ["frequency"]
    df_en_counter = df_en_counter.sort_values(by="frequency", ascending=False)

    return df_en_counter


def gen_new_en(cfg: DictConfig, data: pd.DataFrame) -> pd.DataFrame:
    """Generate DataFrame containing English terms that were detected by polyglot;
    and not listed in mapping dictionaries and combined gaming terms.

    Args:
        cfg (DictConfig):
            OmegaConf dictionary containing configurables for
            gaming related-functions.
        data (pd.DataFrame):
            DataFrame containing English words detected by polyglot
            and its frequency count.

    Returns:
        df_en_new(pd.DataFrame):
            DataFrame containing English terms that were detected by polyglot;
            and not listed in mapping dictionaries and combined gaming terms.
    """

    # Get union of keys for `init_mapping`, `mapping`, `edge_mapping`
    init_mapping_list = list(cfg.init_mapping.keys())
    mapping_list = list(cfg.mapping.keys())
    edge_mapping_list = list(cfg.edge_mapping.keys())
    combined_mapping_list = list(set(init_mapping_list + mapping_list + edge_mapping_list))

    # Get list of gaming terms from `combined_gaming_terms.csv`
    # We lower case these gaming terms since English words detected were lowercased.
    df_combined = pd.read_csv(cfg.paths.combined_path)
    combined_gaming_list = df_combined["term"].str.lower().to_list()

    # Get list of english terms detected
    en_words_list = data["word"].to_list()

    # Remove terms found in combined_mapping_list and combined gaming terms
    en_words_new = list(set(en_words_list) - set(combined_mapping_list) - set(combined_gaming_list))

    print(f"Original number of English words detected : {len(data)}")
    print(f"Number of English words that were not found in combined gaming terms and in mapping : {len(en_words_new)}")
    print(f"Number of English words removed from list : {len(data) - len(en_words_new)}")

    # Convert `en_words_new` list to DataFrame
    df_en_new = pd.DataFrame({"words": en_words_new})

    # Sort by `words` column and append id column
    df_en_new = df_en_new.sort_values(by="words", ascending=True).reset_index(drop=True)
    df_en_new.insert(0, "id", [f"new_en_{i}" for i in df_en_new.index])

    # Saved DataFrame
    df_en_new.to_csv(cfg.paths.gpt_en_new_path)

    return df_en_new


def append_translated(cfg: DictConfig, data: pd.DataFrame) -> pd.DataFrame:
    """Translate unique English words to Malay via Malaya transformer and dictionary.

    Args:
        cfg (DictConfig):
            OmegaConf dictionary containing configurables for
            gaming related-functions.
        data (pd.DataFrame):
            DataFrame containing English terms that were detected by polyglot;
            and not listed in mapping dictionaries and combined gaming terms.

    Returns:
        df (pd.DataFrame):
            DataFrame with appended translated text and word length.
    """

    df = data.copy()

    global TRANSFORMER
    if TRANSFORMER is None:
        # Load Malaya base transformer for translation
        TRANSFORMER = malaya.translation.en_ms.transformer()

    global DICTIONARY
    if DICTIONARY is None:
        # Load Malaya dictionary for translation
        DICTIONARY = malaya.translation.en_ms.dictionary()

    # Append translated text via Malaya dictionary
    df["dictionary"] = df["words"].parallel_map(DICTIONARY.get)

    # Replace missing values with original English text
    df["dictionary"] = df.parallel_apply(lambda row: row["dictionary"] if row["dictionary"] else row["words"], axis=1)

    # Append translated text via Malaya transformer
    df["transformer"] = TRANSFORMER.greedy_decoder(df["words"].to_list())

    # Compute length of English words and its translation
    df["words_len"] = df["words"].str.len()
    df["dictionary_len"] = df["dictionary"].str.len()
    df["transformer_len"] = df["transformer"].str.len()

    # Generate DataFrame for untranslated English words via Malaya dictionary and transformer

    # Save DataFrame
    df.to_csv(cfg.paths.gpt_en_new_path)

    return df


def gen_gpt_audio(
    data: pd.DataFrame,
    col: str,
    audio_dir: str,
    start_index: int = 0,
    batch_size: int = 200,
) -> None:
    """Generate Audio for selected column for GPT corpus.

    Args:
        data (pd.DataFrame):
            DataFrame containing gaming terms and its phonemes.
        col (str):
            Selected column in DataFrame typically either "term" or "norm_term".
        audio_dir (str):
            String containing directory path to audio folder.
        start_index (int):
            Index to start from (Default: 0).
        batch_size (int):
            Batch size (Default: 200).
    """

    # Batch processing
    for index in range(start_index, len(data), batch_size):
        if (index + batch_size) < len(data):
            # column index 0: 'id'
            # column index 5: 'norm_gpt'
            batch = data.iloc[index : index + batch_size, :]
        else:
            batch = data.iloc[index:, :]

        # Generate audio
        gen_audio(batch, col, audio_dir)

        del batch
        gc.collect()

    return None


def modify_path(
    text_file_path: str,
    dest_audio_dir: str,
    new_filename: Optional[str] = None,
) -> None:
    """Modify the file path of the audio output wav to be
    relative to dest_dir in text file.

    Args:
        text_file_path (str):
            String containing file path to text file.
        dest_dir (str):
            String containing directory path to synthesized audio files.
        new_filename (Optional[str]):
            Name of new text file.
    """

    records = []

    # Expand to absolute path
    dest_audio_dir = Path(dest_audio_dir).expanduser().resolve().as_posix()

    logging.info("Reading text file...")

    # Read original text file and modify audio_path
    with open(text_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        logging.info("Changing to actual audio path...")
        for row in reader:
            # row[0] = audio_path; row[1] = text
            text = row[1].strip()

            # Existing audio directory indicated in audio path
            existing_audio_path = Path(row[0]).expanduser().resolve()
            existing_audio_dir = existing_audio_path.parent.as_posix()

            # Amend audio output directory to point to dest_audio_dir
            output_path = row[0].replace(
                existing_audio_dir,
                dest_audio_dir,
            )

            records.append((output_path, text))

    logging.info("Existing audio directory: %s", existing_audio_dir)
    logging.info("Actual audio directory : %s", dest_audio_dir)
    logging.info("Successfully amended audio path.")

    if new_filename:
        # Generate absolute file path with new filename
        new_file_path = Path(text_file_path).with_name(new_filename).as_posix()
        write_to_csv(new_file_path, records)
        logging.info("Successfully save updated text file as %s.", new_filename)

    else:
        write_to_csv(text_file_path, records)
        logging.info("Successfully update existing text file.")

    return None


def write_to_csv(csv_file: str, records: List[Tuple[str, str]], headers: Optional[List[str]] = None) -> None:
    """Generate csv file in text format.

    Args:
        text_file (str):
            String containing file path to text file used as input to vits training.
        records (List[Tuple[str, str]]):
            List of tuples containing audio path and phonemes.
        headers (Optional[List[str]]):
            Optional list containing column names for csv file.

    Returns:
        None
    """

    # Create folder to store csv file if not exist
    csv_dir = Path(csv_file).parent

    if not csv_dir.is_dir():
        csv_dir.mkdir(parents=True, exist_ok=False)

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter="|")

        # Write header row if headers is not None
        if headers:
            writer.writerow(headers)

        # Write multiple rows of reocrds
        writer.writerows(records)

    return None


def gen_gpt_stats(gpt_txt_path: str) -> pd.DataFrame:
    """Generate DataFrame contains audio and textual information for each
    text-audio pair

    Args:
        gpt_txt_path (str):
            Absolute path to `gpt.txt` required for VITS inferencing.

    Returns:
        df_stats (pd.DataFrame):
            DataFrame containing audio and textual information for each text-audio pair.
        df_word_count (pd.DataFrame):
            DataFrame containing count of words that are found in gpt corpus.
        df_punct_count (pd.DataFrame):
            DataFrame containing count of punctuations that are found in gpt corpus.
        desc_stats (pd.DataFrame):
            DataFrame containing descriptive statistics of df_stats,
            df_word_count and df_punct_count combined into one.
        summary_stats (pd.DataFrame):
            DataFrame containing total text_len, total word_count and total duration.
    """

    # Initialize dictionary to store required audio and textual information
    gpt_dict = {
        "id": [],
        "audio_path": [],
        "text": [],
        "text_len": [],
        "word_count": [],
        "duration": [],
    }

    # Initialize counter for word and punctuation count
    word_counter = Counter()
    punct_counter = Counter()

    # Read `gpt.txt` file in csv format
    with open(gpt_txt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")

        for row in reader:
            # Get id, audio path and text transcript
            audio_path = Path(row[0])
            id = audio_path.stem
            text = row[1]

            # Update gpt_dict on id, audio path and text transcript
            gpt_dict["id"].append(id)
            gpt_dict["audio_path"].append(audio_path.as_posix())
            gpt_dict["text"].append(text)

            # text_len = Number of characters in text transcript including white space
            gpt_dict["text_len"].append(len(text))

            # Number of words in text transcript where word is string of alphanumeric characters separated by white space
            word_list = [word for word in text.split(" ") if word not in string.punctuation]
            punct_list = [word for word in text.split(" ") if word in string.punctuation]

            # word_count = Number of words in transcript
            gpt_dict["word_count"].append(len(word_list))

            # Capture number of times a word and punctuation appears in dataset
            word_counter.update(word_list)
            punct_counter.update(punct_list)

            # Generate audio wav file if not exist
            if not audio_path.is_file():
                print(f"Missing audio for id: {id}\nCorresponding text transcript: {text}")
                gen_wav_file(text, audio_path)

            # Update gpt_dict with duration for audio file
            duration = parselmouth.Sound(audio_path.as_posix()).duration
            gpt_dict["duration"].append(duration)

    # Generate df_stats
    df_stats = pd.DataFrame(gpt_dict)

    # Generate df_word_count
    df_word_count = counter_to_DataFrame(word_counter, ["word", "unique_word_count"])
    df_word_count["num_chars_per_word"] = df_word_count["word"].str.len()

    # Generate df_punct_count
    df_punct_count = counter_to_DataFrame(punct_counter, ["punctuation", "unique_punct_count"])

    # Generate summary_stats and desc_stats
    (
        desc_stats,
        summary_stats,
    ) = summary_gpt_stats(df_stats, df_word_count, df_punct_count)

    return (df_stats, df_word_count, df_punct_count, desc_stats, summary_stats)


def counter_to_DataFrame(counter_obj: Counter, col_list: List[str]) -> pd.DataFrame:
    """Convert counter object to DataFrame with 2 columns i.e.
    counter_obj.keys and counter_obj.values

    Args:
        counter_obj (Counter): Counter object to be converted to DataFrame.
        col_list [List[str]]: List of column names

    Returns:
        pd.DataFrame: DataFrame converted from Counter object.
    """

    # Convert counter object to list of list containing key and value
    counter_list = [[k, v] for k, v in counter_obj.items()]

    return pd.DataFrame(counter_list, columns=col_list)


def summary_gpt_stats(df_stats: pd.DataFrame, df_word: pd.DataFrame, df_punct: pd.DataFrame) -> None:
    """Display summary statistics for `gpt.txt`.

    Args:
        df_stats (pd.DataFrame):
            DataFrame containing audio and textual information for each text-audio pair.
        df_word (pd.DataFrame):
            DataFrame containing count of words that are found in gpt corpus.
        df_punct (pd.DataFrame):
            DataFrame containing count of punctuations that are found in gpt corpus.

    Returns:
        desc_stats (pd.DataFrame):
            DataFrame containing descriptive statistics of df_stats,
            df_word and df_punct combined into one.
        summary_stats (pd.DataFrame):
            DataFrame containing total text_len, total word_count and total duration.
    """

    # Display descriptive statistics
    desc_stats = pd.concat([df_stats.describe(), df_word.describe(), df_punct.describe()], axis=1)

    # summary statistics
    summary_stats = {col: df_stats[col].sum() for col in ["text_len", "word_count", "duration"]}

    return desc_stats, pd.DataFrame.from_dict(summary_stats, orient="index", columns=["Total Value"])


def display_yaml(cfg: DictConfig, data: pd.DataFrame, key_col: str = "words", value_col: str = "malay_word") -> None:
    """Display key-value pair in yaml format.

    Args:
        cfg (DictConfig):
            OmegaConf dictionary containing required mapping to normalize English text.
        data (pd.DataFrame):
            DataFrame containing English words and its corresponding translated words in Malay.
        gaming_list (List[str]):
            List of combined gaming terms.
        key_col (str, optional):
            Column name containing English words (Default: "words").
        value_col (str, optional):
            Column name containing translated words in Malay (Default: "malay_word").

    Returns:
        None
    """

    # Obtain gaming list
    df_combined = pd.read_csv(cfg.paths.combined_path)
    gaming_list = df_combined["term"].to_list()

    for row in data.loc[:, ["words", "malay_word"]].itertuples(index=False, name=None):
        # Extract words that are different from its translation and not found in gaming list
        # and existing mapping.
        if (
            row[0] != row[1]
            and len([word for word in gaming_list if row[0] in word]) == 0
            and len([row[0] for k, v in cfg.mapping.items() if row[0] in v or row[0] in k]) == 0
        ):
            # Remove aprostophe and replace hyphen with white space
            word = row[0].replace("-", " ")
            word = word.replace("'", "")
            malay_word = row[1].replace("-", " ")

            print(f"  {word.lower().strip()}: {malay_word.lower().strip()}")
