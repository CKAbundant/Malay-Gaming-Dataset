"""Utilities functions required for generating phonemes for synthetic
text-audio pair based on ChatGPT 3.5 corpus.
"""

import csv
import gc
import json
import logging
import re
import string
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import eng_to_ipa
import malaya
import pandas as pd
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

# Initialize PHON_MALAY
cfg_gpt = OmegaConf.load(GPT_YAML_PATH)
PHON_MALAY = list(cfg_gpt.phon_malay)


def initialize_pandarallel(progress_bar, num_proc: int):
    """Initialize pandarallel"""
    return pandarallel.initialize(progress_bar=progress_bar, nb_workers=num_proc)


def normalize_gpt_phon(text: str, yaml_path: str) -> str:
    """Normalize gpt text based on mapping catered to eng_to_ipa phonemizer.

    Args:
        text (str): Text string to be normalized.
        yaml_path (str): Absolute path to `gaming_gpt.yaml`

    Returns:
        norm_text (str): Normalized gaming term.
    """

    # Load mapping from gaming_gpt.yaml
    cfg = OmegaConf.load(yaml_path)
    mapping = cfg.mapping
    custom_mapping = cfg.custom_mapping
    translated_mapping = cfg.translated_mapping
    edge_mapping = cfg.edge_mapping

    # Normalize edge cases
    for term, replacement in edge_mapping.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, replacement, text)

    # Replace slash, hyphen and brackets with white space
    # Replace apostrophe
    text = pre_clean_text(text)

    for term, replacement in mapping.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

    for term, replacement in custom_mapping.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

    for term, replacement in translated_mapping.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

    return remove_extra_whitespace(text)


def phon_eng2ipa(text: str) -> str:
    """Phonemize GPT corpus via eng_to_ipa.

    Args:
        text (str):
            Normalized GPT text for phonemization i.e. taken from
            `norm_phon` in `gpt.csv`.

    Returns:
        (str):
            Phonemized text using eng_to_ipa only.

    Note:
        eng_to_ipa would remove all punctuations if applied directly
        on text transcript.
    """

    # Pre-processing similar to malay_cleaners2
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # phon_list = [
    #     word.strip() if word in string.punctuation or word in PHON_MALAY else word.strip() for word in text.split(" ")
    # ]
    phon_list = []

    for word in text.split(" "):
        word = word.strip()

        if word in PHON_MALAY:
            # Malay word that is incorrectly phonemized by eng_to_ipa
            # Tag asterick at end
            phon_list.append(f"{word}*")

        elif word in string.punctuation:
            # Append punctuations without phonemizing by eng_to_ipa
            phon_list.append(word)

        else:
            # Phonemize by eng_to_ipa
            phon_list.append(eng_to_ipa.convert(word))

    # Convert phoneme list to string
    phon_string = (" ").join(phon_list)

    return remove_extra_whitespace(phon_string)


def phon_espeak_ms(text: str) -> str:
    """Phonemize GPT corpus via espeak-ms.

    Args:
        text (str):
            Normalized GPT text for phonemization i.e. taken from
            `norm_phon` in `gpt.csv`.

    Returns:
        (str):
            Phonemized text using espeak-ms only.
    """

    global BACKEND_MS

    # Initialize espeak-en and espeak-ms backend
    if BACKEND_MS is None:
        BACKEND_MS = EspeakBackend(
            language="ms", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    # Pre-processing similar to malay_cleaners2
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Words that are not phonemized by eng_to_ipa is tagged with asterick;
    # and phonemized with espeak-ms. Words that can be phonemized by eng_to_ipa
    # is phonemized by espeak-en instead. We are using backend instead of
    # original english_cleaners2 and malay_cleaners2 because of memory concern.
    phon_list = BACKEND_MS.phonemize([text], strip=True)

    # phon_list contans only 1 string
    return remove_extra_whitespace(phon_list[0])


def phon_espeak_en_ms(text: str, phon_mapping: DictConfig) -> str:
    """Phonemize English words by espeak-en and Malay words by espeak-ms.

    Args:
        text (str):
            Normalized GPT text for phonemization i.e. taken from
            `norm_phon` in `gpt.csv`.
        phon_mapping (DictConfig):
            OmegaConf dictionary mapping English words to phonemes
            based on CMU English vocabulary.

    Returns:
        (str):
            Phonemized text using espeak-en and espeak-ms.
    """

    global BACKEND_MS, BACKEND_EN

    # Initialize espeak-en and espeak-ms backend
    if BACKEND_MS is None:
        BACKEND_MS = EspeakBackend(
            language="ms", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    if BACKEND_EN is None:
        BACKEND_EN = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    # Pre-processing similar to malay_cleaners2 and english_cleaners2
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Words that are not phonemized by eng_to_ipa is tagged with asterick;
    # and phonemized with espeak-ms. Words that can be phonemized by eng_to_ipa
    # is phonemized by espeak-en instead. We are using backend instead of
    # original english_cleaners2 and malay_cleaners2 because of memory concern.
    phon_list = []

    for word in text.split(" "):
        word = word.strip()

        if word in phon_mapping:
            phon_list.extend(BACKEND_EN.phonemize([word], strip=True))

        elif eng_to_ipa.convert(word).endswith("*") or word in PHON_MALAY:
            # Phonemize non-English words with espeak-ms
            phon_list.extend(BACKEND_MS.phonemize([word], strip=True))

        else:
            # Phonemize English words with espeak-en
            phon_list.extend(BACKEND_EN.phonemize([word], strip=True))

    # Convert phoneme list to string
    phon_string = (" ").join(phon_list)

    return remove_extra_whitespace(phon_string)


def phon_eng2ipa_espeak(text: str, phon_mapping: DictConfig) -> str:
    """Phonemize English words by eng_to_ipa and Malay words by espeak-ms.

    Args:
        text (str):
            Normalized GPT text for phonemization i.e. taken from
            `norm_phon` in `gpt.csv`.
        phon_mapping (DictConfig):
            OmegaConf dictionary mapping English words to phonemes
            based on CMU English vocabulary.

    Returns:
        (str):
            Phonemized text using eng_to_ipa and espeak-ms.
    """

    global BACKEND_MS

    # Initialize espeak-en and espeak-ms backend
    if BACKEND_MS is None:
        BACKEND_MS = EspeakBackend(
            language="ms", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    # Pre-processing similar to malay_cleaners2 and english_cleaners2
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Words that are not phonemized by eng_to_ipa is tagged with asterick;
    # and phonemized with espeak-ms. Words that can be phonemized by eng_to_ipa
    # is phonemized by espeak-en instead.

    phon_list = []

    for word in text.split(" "):
        word = word.strip()
        phon_text = eng_to_ipa.convert(word)

        # Words not able to be phonemized by eng_to_ipa
        if phon_text.endswith("*"):
            try:
                # Attempt to phonemize via `phon_mapping`
                phon_list.append(phon_mapping[word])
            except:
                # Words not found in `phon_mapping` are likely Malay words
                phon_list.extend(BACKEND_MS.phonemize([word], strip=True))

        elif word in PHON_MALAY:
            # Phonemize Malay words that are wrongly identified by eng_to_ipa
            # via espeak-en
            phon_list.extend(BACKEND_MS.phonemize([word], strip=True))

        elif word in string.punctuation:
            # Append punctuation marks without phonemizing
            phon_list.append(word)

        else:
            # Append phonemes generated via eng_to_ipa
            phon_list.append(phon_text)

    # Convert phoneme list to string
    phon_string = (" ").join(phon_list)

    return remove_extra_whitespace(phon_string)


def phonemize_gpt(
    data: pd.DataFrame, col: str, text_col: str = "norm_phon", phon_mapping: Optional[DictConfig] = None
) -> pd.DataFrame:
    """Phonemize normalized GPT corpus based on column input.

    Args:
        data (pd.DataFrame):
            DataFrame containing normalized GPT corpus for phonemization.
        col (str):
            Either "eng2ipa", "espeak_ms", "espeak_en_ms" or "eng2ipa_espeak_ms".
        text_col (str):
            Name of column containing the normalized text for phonemization
            (Default: "norm_phon").
        phon_mapping (Optional[DictConfig]):
            OmegaConf dictionary mapping English words to phonemes
            based on CMU English vocabulary.

    Returns:
        df (pd.DataFrame):
            DataFrame appended with phonemized GPT corpus.
    """

    global BACKEND_MS, BACKEND_EN
    df = data.copy()

    # Initialize espeak-en and espeak-ms backend
    if BACKEND_MS is None:
        BACKEND_MS = EspeakBackend(
            language="ms", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    if BACKEND_EN is None:
        BACKEND_EN = EspeakBackend(
            language="en-us", preserve_punctuation=True, with_stress=True, language_switch="remove-flags"
        )

    # Start timing after espeak backend are loaded
    start_time = time.perf_counter()

    if col == "eng2ipa":
        df[col] = df[text_col].parallel_map(phon_eng2ipa)

    elif col == "espeak_ms":
        df[col] = df[text_col].parallel_map(phon_espeak_ms)

    elif col == "espeak_en_ms":
        df[col] = df[text_col].parallel_map(lambda x: phon_espeak_en_ms(x, phon_mapping))

    else:
        df[col] = df[text_col].parallel_map(lambda x: phon_eng2ipa_espeak(x, phon_mapping))

    end_time = time.perf_counter()
    print(f"\nTotal time taken : {end_time - start_time} seconds")

    return df


def expand_compound(text_list: List[str]) -> pd.DataFrame:
    """Expand list of compound words and convert to DataFrame.

    Args:
        text_list (List[str]): List containing compound words.

    Returns:
        df (pd.DataFrame): DataFrame containing compound words.
    """

    expand_compound = []

    for word in text_list:
        # Split words based on hyphen or slash
        expand_word = re.split(r"[-/\s]", word)

        # Extend list with lowercased non-empty term
        expand_compound.extend([term.strip().lower() for term in expand_word if term])

    # Convert to DataFrame
    df = pd.DataFrame({"words": expand_compound})

    # Remove any duplicates
    df = df.drop_duplicates(subset="words")

    return df


def create_dict(gpt_dict: Dict[str, str], start_word: str = "abilities") -> Tuple[Dict[str, str]]:
    """Split `gpt_mapping` dictionary into `custom_dict` and `translated_dict`
    given     the starting word for translated key-value pair. Note that
    `gpt_mapping` in `gaming.yaml`     is structured by having the custom
    key-value pair followed by translated key-value pair.

    Args:
        gpt_dict (Dict[str, str]):
            Dictionary converted from OmegaConf Dictionary `gpt_mapping`.
        start_word (str, optional):
            First word in translated key-value pair (Defaults: 'abilities').

    Returns:
        custom_dict (Dict[str, str]):
            Dictionary mapping English words to modified forms.
        translated_dict (Dict[str, str]):
            Dictionary mapping English words to translated Malay words.
    """

    # Convert gpt_mapping.keys to list
    gpt_dict_keys = list(gpt_dict.keys())

    # Get index for 'abilities'
    start_index = gpt_dict_keys.index("abilities")

    # Split gpt_mapping_keys into 2 lists i.e. custom_list and translated_list
    custom_list = sorted(gpt_dict_keys[:start_index])
    translated_list = sorted(gpt_dict_keys[start_index:])

    # Generate `custom_mapping` and `translated_mapping` dictionary
    custom_dict = {key: gpt_dict[key] for key in custom_list}
    translated_dict = {key: gpt_dict[key] for key in translated_list}

    return custom_dict, translated_dict


def extract_phon_from_mapping(data: Dict[str, str], choice: str = "phon") -> pd.DataFrame:
    """Extract list of phonemized words (via eng_to_ipa) from
    `mapping` in `gaming.yaml`.

    Args:
        data (Dict[str, str]):
            Dictionary mapping words to modified words for
            Malaya VITS model inferencing.
        choice (str):
            Either "phon" or "unphon" (Default: "phon").

    Returns:
        (pd.DataFrame):
            DataFrame containing phonemized keys belonging to `mapping_dict`.
    """

    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")

    # Reset index and set column names
    df = df.reset_index()
    df.columns = ["words", "mapping"]

    # Apply eng_to_ipa to words
    df["eng2ipa"] = df["words"].parallel_map(eng_to_ipa.convert)

    if choice == "unphon":
        # Extract list of phonemized words
        return df.loc[df["eng2ipa"].str.endswith("*"), :]

    return df.loc[~df["eng2ipa"].str.endswith("*"), :]


def keys_in_values(translated_dict: Dict[str, str], mapping_dict: Dict[str, str]) -> List[str]:
    """Generate list of keys in translated_dict that are found in values of mapping_dict.

    Args:
        translated_dict (Dict[str, str]):
            Dictionary mapping English words to its translation in Malay.
        mapping_dict (Dict[str, str]):
            Dictionary mapping gaming terms to its normalized version
            i.e. combination of English words.

    Returns:
        List[str]:
            List containing keys in translated_dict that are found in
            values of mapping_dict.
    """

    remove_key = []

    for key in translated_dict.keys():
        # for each key in translated_dict, iterate through all value
        # in mapping_dict to check. If found, then break inner loop.
        for value in mapping_dict.values():
            if key in value:
                remove_key.append(key)
                break

    return remove_key


def keys_in_keys(dict1: Dict[str, str], dict2: Dict[str, str]) -> List[str]:
    """Generate list of keys in translated_dict that are found in values of mapping_dict.

    Args:
        dict1 (Dict[str, str]): Dictionary to remove common keys.
        dict2 (Dict[str, str]): Dictionary to compare with dict1.

    Returns:
        List[str]:
            List containing keys in translated_dict that are found in
            values of mapping_dict.
    """

    remove_key = []

    for key1 in dict1.keys():
        # for each key in dict1, iterate through all keys in dict2.
        # If found, then break inner loop.
        for key2 in dict2.keys():
            if key1.lower() in key2.lower().split(" "):
                remove_key.append(key1)
                break

    return remove_key


def save_json(data: Dict[str, str], file_path: str) -> None:
    """Save dictionary as json file at designated file_path.

    Args:
        data (Dict[str, str]): Dictionary to be saved as json file.
        file_path (str): Absolute path to json file.
    """

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def gen_unphon_gaming(combined_path: str) -> List[str]:
    """Generate list of unphonemized gaming terms from `combined_gaming_terms.csv`.

    Args:
        combined_path (str): Absolute path to `combined_gaming_terms.csv`.

    Returns:
        (List[str]): List of unphonemized gaming terms.
    """

    # load `combined_gaming_terms.csv` as DataFrame
    df_combined = pd.read_csv(combined_path)

    # Extract unphonemized gaming terms
    df_unphon_gaming = df_combined.loc[df_combined["eng2ipa"].str.endswith("*"), ["term", "eng2ipa"]]

    # Remove hyphen
    df_unphon_gaming["term"] = df_unphon_gaming["term"].str.replace("-", " ", regex=False)

    # Lower case gaming terms
    df_unphon_gaming["term"] = df_unphon_gaming["term"].str.lower()

    return df_unphon_gaming["term"].to_list()


def extract_phon_unphon(data: pd.DataFrame, choice: str = "phon", phon_col: str = "eng2ipa") -> pd.DataFrame:
    """Extract phonemized or unphonemized words by eng_to_ipa in DataFrame.

    Args:
        data (pd.DataFrame):
            DataFrame containing phonemized words by eng_to_ipa.
        choice (str):
            Either "phon" or "unphon" (Default: "phon").
        phon_col (str):
            Name of column containing eng_to_ipa phonemes (Default: "eng2ipa").

    Returns:
        (pd.DataFrame):
            DataFrame containing either phonemized or unphonemized words.
    """

    if choice == "unphon":
        return data.loc[data[phon_col].str.contains("*", regex=False), :]

    return data.loc[~data[phon_col].str.contains("*", regex=False), :]


def check_word(data: pd.DataFrame, word: str, col: str = "words") -> pd.DataFrame:
    """Check if word is present in the selected column of DataFrame.

    Args:
        data (pd.DataFrame):
            DataFrame contanining text of interest.
        word (str):
            Word to be checked against DataFrame.
        col (str, optional):
            Name of column containing text of interest (Default: "words").

    Returns:
        (pd.DataFrame): Filtered DataFrame containing word of interest.
    """

    return data.loc[data[col].str.contains(word, case=False), :]


def extract_unique(listings: List[List[str]]) -> List[str]:
    """Extract unique items that are found in first list in listings but
    not in the rest of the listings.

    Args:
        listings (List[List[str]]):
            List containing list of strings to compare.

    Returns:
        unique_list (List[str]):
            List of unique items that are found in list1 but not in list2.
    """

    temp = {}
    for i in range(len(listings)):
        if i == 0:
            temp = set(listings[i])
        else:
            temp = temp - set(listings[i])

    print(len(temp))

    return list(temp)


def extract_non_english(data: pd.DataFrame) -> pd.DataFrame:
    """Extract non english words from entire GPT corpus.

    Args:
        data (pd.DataFrame): DataFrame containing GPT corpus.

    Returns:
        df1 (pd.DataFrame): DataFrame containing unphonemized words.
    """

    # Set word counter
    word_counter = Counter()

    # Iterate through GPT corpus and perform word count
    for text in data["text"].to_list():
        # Only update words and not punctuation
        word_counter.update([word.strip().lower() for word in text.split(" ") if word not in string.punctuation])

    print(len(word_counter))

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(word_counter, orient="index").reset_index()
    df.columns = ["words", "frequency"]

    # Apply eng_to_ipa
    df["eng2ipa"] = df["words"].parallel_map(eng_to_ipa.convert)

    # Select phonemized words
    df1 = extract_phon_unphon(df)

    # Drop duplicates
    df1 = df1.drop_duplicates(subset="words")

    # Sort by frequency in descending order
    df1 = df1.sort_values(by="frequency", ascending=False).reset_index(drop=True)

    for row in df1.itertuples(index=False, name=None):
        print(f'    "{row[0]}",')

    return df1


def compute_word_count(text: str) -> int:
    """Compute number of words excluding punctuation marks.

    Args:
        text (str): Text string.

    Returns:
        int: Number of owrds in text string.
    """

    return len([word for word in text.split(" ") if word not in string.punctuation])


def common_translated_gaming(combined_path: str, translated_dict: Dict[str, str]) -> List[str]:
    """Identify common terms in `translated_dict` and expanded gaming terms.

    Args:
        combined_path (str):
            Absolute path to `combined_gaming_terms.csv`.
        translated_dict (Dict[str, str]):
            Dictionary mapping English words to equivalent Malay words

    Returns:
        remove_list (List[str]):
            List of common terms in `translated_dict` and expanded gaming terms.
    """

    remove_list = []

    # Load `combined_gaming_terms.csv` as DataFrame
    df = pd.read_csv(combined_path)

    # Iterate through keys in expand_keys to check if it appears in gaming_list
    # Break inner loop if key is found in gaming_list
    for key in translated_dict.keys():
        for word in df["term"].to_list():
            # Replace slash, hyphen and brackets with white space
            # Replace apostrophe
            word = pre_clean_text(word)

            # Break loop if key is found in word
            pattern = rf"^{re.escape(key)}$|^{re.escape(key)} | {re.escape(key)}$"
            if len(re.findall(pattern, word, flags=re.IGNORECASE)) > 0:
                remove_list.append(key)
                break

    return remove_list


def expand_words(text_list: List[str]) -> List[str]:
    """Expand text string in list into list of words.

    Args:
        text_list (List[str]): List of text string containing multiple words.

    Returns:
        word_list (List[str]): List of words.
    """

    # Extract gaming terms and expand out compound gaming terms as list
    # Replace hyphen with white space
    # Example: "always on drm" -> "always", "on", "drm"
    word_list = [word.replace("-", " ").lower() for text in text_list for word in text.split(" ") if word]

    # Remove duplicated terms and sort by alphabetical order
    word_list = sorted(list(set(word_list)))

    return word_list
