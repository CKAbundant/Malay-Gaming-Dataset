"""Utilities functions for normalization and phonemization
of gaming terms.
"""

import csv
import gc
import logging
import re
import string
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import eng_to_ipa
import IPython.display as ipd
import malaya_speech
import pandas as pd
import requests
from bs4 import BeautifulSoup
from omegaconf import DictConfig, OmegaConf, open_dict
from pandarallel import pandarallel
from polyglot.detect import Detector
from polyglot.detect.base import logger as polyglot_logger
from tqdm import tqdm
from unidecode import unidecode

# Append repo root directory to system path
polyglot_logger.setLevel("ERROR")
tqdm.pandas()

from notebooks.src_eval.metrics_gen_utils import compute_alignment_score, compute_cer

from ..src_eval.batch_inferencing_vits import vits_batch_infer
from .eval_phonemes import eval_phon
from .text.cleaners import english_cleaners2, malay_cleaners2
from .text_processing import LETTERS_IPA, symbols

URL = "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms"
YAML_PATH = Path(__file__).parents[2].joinpath("notebooks", "src_gaming", "gaming.yaml")
VITS_MODEL = None
WHISPER_MODEL = None


def initialize_pandarallel(progress_bar, num_proc: int):
    """Initialize pandarallel"""
    return pandarallel.initialize(progress_bar=progress_bar, nb_workers=num_proc)


def compare_oov(
    data: pd.DataFrame, cols: List[str], file_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Generate DataFrame that contains out-of-vocab (oov) words
    that are not phonemized properly.

    Args:
        data (pd.DataFrame):
            DataFrame containing gaming terms and corresponding IPA phonemes.
        cols (List[str]):
            List of columns to compare.
        file_path (Optional[str]):
            If provided, save df_stats as csv file in path provided by file_path.

    Returns:
        df_stats (pd.DataFrame):
            DataFrame containing oov statistics of different phonemizers.
        df_dict (Dict[str, pd.DataFrame]):
            Dictionary contanining oov DataFrame for different phonemizers.
    """

    oov_dict = {col: [] for col in ["phon_type", "num_gaming", "num_oov", "percent_oov"]}
    df_dict = {}

    for col in cols:
        # Select rows where 'ipa' column contains "*" character
        df_oov = data.loc[data[col].str.contains(pat="*", regex=False), :].reset_index(drop=True)
        df_dict[col] = df_oov

        # Calculate number of oov and percentage of oov
        num_oov = len(df_oov)
        percent_oov = round(num_oov / len(data) * 100, 2)

        # Update oov_dict
        oov_dict["phon_type"].append(col)
        oov_dict["num_gaming"].append(len(data))
        oov_dict["num_oov"].append(num_oov)
        oov_dict["percent_oov"].append(percent_oov)

    # Generate DataFrame containing oov statistics
    df_stats = pd.DataFrame(oov_dict)

    if file_path:
        df_stats.to_csv(file_path, index=False)

    return df_stats, df_dict


def download_gaming_terms(
    url: str = URL,
) -> List[str]:
    """Download gaming terms from wikipedia url to a list.

    Args:
        url (str):
            URL to gaming terms wikpedia
            (Defaults: "https://en.wikipedia.org/wiki/Glossary_of_video_game_terms").

    Returns:
        (List[str]):
            List containing gaming terms from wikipedia.
    """

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract words within <dfn class='glossary'> tags
    glossary_terms = soup.find_all("dfn", class_="glossary")

    # Extract and normalize the glossary terms
    return [term.get_text() for term in glossary_terms]


def expand_terms(gaming_terms: List[str]) -> List[str]:
    """Expand multiple gaming terms e.g.
    'action role-playing game (ARPG)' -> ['action role-playing game', 'ARPG']

    Args:
        gaming_terms (List[str]): List of gaming terms.

    Returns:
        List[str]: List of gaming terms with multiple terms expanded.
    """

    pattern = r"\s*\(([^)]*)\)\s*"
    expanded_list = []

    for gaming_term in gaming_terms:
        # Strip white spaces first
        gaming_term = gaming_term.strip()

        # Split gaming term by slash character only for "pog/poggers"
        if gaming_term in ["pog/poggers", "clock/clocked"]:
            expanded_list.extend(gaming_term.split("/"))

        else:
            # Expand term encapsulated by brackets
            for term in re.split(pattern, gaming_term):
                # Remove white spaces and double or single quotes
                term = re.sub(r"[\"\']", "", term).strip()

                # Remove "or " if item starts with "or "
                if term.startswith("or "):
                    term = term.replace("or ", "")

                # Split item into list if contains commas
                # Extend expanded_list
                expanded_list.extend([item.strip() for item in re.split(r",| or ", term) if item])

    return expanded_list


def append_ipa(
    data: pd.DataFrame,
    col_name: str,
    file_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate DataFrame containing gaming terms and its
    corresponding IPA phonemes.

    Args:
        data (pd.DataFrame):
            Data Frame containing both unnormalized and normalized
            gaming terms web-scrapped from wikipedia.
        col_name (str):
            Column name to be assigned to phonemes.
        file_path (str):
            If provided, appended DataFrame will be saved to file_path.

    Returns:
        (List[str]):
            List of phonemes.
    """

    mapping = {
        "eng2ipa": ["term", eng_to_ipa.convert],
        "eng2ipa_n": ["norm_term", eng_to_ipa.convert],
        "ipa_en": ["term", english_cleaners2],
        "ipa_norm_en": ["norm_term", english_cleaners2],
        "ipa_ms": ["term", malay_cleaners2],
        "ipa_norm_ms": ["norm_term", malay_cleaners2],
        "en_en_un": ["ipa_en", english_cleaners2],
        "ms_en_un": ["ipa_en", malay_cleaners2],
        "en_en_n": ["ipa_norm_en", english_cleaners2],
        "ms_en_n": ["ipa_norm_en", malay_cleaners2],
    }

    # Append IPA phonemes
    data[col_name] = data[mapping[col_name][0]].parallel_map(mapping[col_name][1])

    # Save DataFrame if file_path provided
    if file_path:
        data.to_csv(file_path, index=False)

    return data


def pre_clean_text(text: str) -> str:
    """Replace slash, comma, brackets and hyphen with whitespace

    Args:
        text (str): Text string.

    Returns:
        (str): Text string with slash and hyphen replaced.
    """

    # Replace slash, brackets, and hyphen with whitespace
    text = re.sub(r"[-/\(\)]", " ", text)

    # Replace apostrophe
    text = re.sub("'", "", text)

    return text


def remove_extra_whitespace(text: str) -> str:
    """Remove extra whitespace.

    Args:
        text (str): Text string.

    Returns:
        str: Text string with extra whitespaces removed.
    """

    norm_text = re.sub(r"\s\s+", " ", text)

    return norm_text.strip()


def gen_df(gaming_list: List[str], choice: str = "init", file_path: Optional[str] = None) -> pd.DataFrame:
    """Generate DataFrame containing gaming terms and normalized terms.

    Args:
        gaming_list (List[str]):
            List of gaming terms.
        choice (str):
            Either "init" i.e. actual english words to normalized or
            "custom" i.e. manually manipulate characters to generate audio
            (Default: "init").
        file_path (Optional[str]):
            If provided, generated DataFrame will be saved to file_path.

    Returns:
        (pd.DataFrame):
            DataFrame containing gaming terms and its phonemes.
    """

    # Convert gaming_list into DataFrame
    df = pd.DataFrame({"term": gaming_list})

    # Generate and append normalized gaming terms
    df["norm_term"] = df["term"].parallel_map(lambda x: normalize_gaming(x, choice))

    # Apply customized phomeization

    # Sort by term
    df = df.sort_values(by="term", ascending=True).reset_index(drop=True)

    # Remove duplicated term
    dup = df.duplicated().sum()
    df = df.drop_duplicates(subset=["term"]).reset_index(drop=True)
    print(f"Number of duplicated terms detected and removed: {dup}")

    # Insert id column
    df.insert(0, "id", df.index)

    # Create directory if non existiing
    dir_path = Path(file_path).parent

    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)

    if file_path:
        df.to_csv(file_path, index=False)

    return df


def normalize_gaming(text: str, choice: str = "custom") -> str:
    """Normalize gaming terms based on custom mapping

    Args:
        text (str):
            Text string to be normalized.
        choice (str):
            Either "init" i.e. actual english words to normalized or
            "custom" i.e. manually manipulate characters to generate audio
            (Default: "custom").

    Returns:
        norm_text (str): Normalized gaming term.
    """

    # Load mapping from gaming.yaml
    cfg = OmegaConf.load(YAML_PATH)
    MAPPING = cfg.mapping if choice == "custom" else cfg.init_mapping
    EDGE_MAPPING = cfg.edge_mapping

    # Normalize edge cases
    for term, replacement in EDGE_MAPPING.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, replacement, text)

    # Replace slash, hyphen and brackets with white space
    text = pre_clean_text(text)

    for term, replacement in MAPPING.items():
        pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"
        text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

    return remove_extra_whitespace(text)


def phonemize_gaming(text: str) -> str:
    """Phonemize normalized gaming terms via eng_to_ipa based on `phon_mapping`.

    Args:
        text (str): Text string to be phonemized via eng_to_ipa.

    Returns:
        (str): Phonemized gaming term.
    """

    # Load mapping from gaming.yaml
    cfg = OmegaConf.load(YAML_PATH)
    mapping = cfg.phon_mapping

    if "*" in text:
        for term, replacement in mapping.items():
            # Unphonemized text ends with *
            term = f"{term}*"

            # Pattern to capture standalone term, term at start, end and inbetween
            pattern = rf"^{re.escape(term)}$|\s+{re.escape(term)}\s+|^{re.escape(term)} | {re.escape(term)}$"

            # Replace text with replacement regardless of casing for text
            text = re.sub(pattern, f" {replacement} ", text, flags=re.IGNORECASE)

        return remove_extra_whitespace(text)

    return text


def read_additional(file_path: str, save_file: bool = False) -> List[str]:
    """Read sorted additional gaming terms written in additional_term.txt.

    Args:
        file_path (str):
            String containing file path to "additional_term.txt".
        save_file (bool):
            Whether to save the updated sorted list (Default: False).

    Returns:
        dataList[str]:
            List containing additional gaming terms to be updated
    """

    # Read additional_term.txt
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        data = [term.strip() for term in data]

    # Remove duplicates and sort list
    data_clean = list(set(data))
    data_clean = sorted(data_clean)

    if save_file:
        write_to_txt(data_clean, file_path)

    return data_clean


def write_to_txt(data: List[str], file_path: str) -> None:
    """Write list of strings to text file.

    Args:
        data (List[str]): List of text strings.
        file_path (str): String containing file path to text file.

    Returns:
        None
    """

    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")

    return None


def update_gaming_terms(pth: DictConfig) -> pd.DataFrame:
    """Check for duplicates in additional terms. Process and append
    new terms to DataFrame.

    Args:
        pth (DictConfig):
            Omegaconfig dictionary containing required file paths.

    Returns:
        df_combined (pd.DataFrame):
            DataFrame with appended additional gaming terms.
    """

    # Read gaming_term.csv and additional_term.txt
    df_gaming = pd.read_csv(pth.gaming_path)
    additional_terms = read_additional(pth.additional_path)

    # Strip excess white spaces and lowercase gaming_terms from gaming_terms.csv
    gaming_list = df_gaming["term"].map(lambda x: x.lower().strip()).to_list()

    # Extract non-duplicate terms from additional_terms
    updated_terms = [term for term in additional_terms if term.lower().strip() not in gaming_list]

    # Generate DataFrame with required ipa phonemes from updated_terms
    df_updated = gen_df(updated_terms, file_path=pth.updated_path)
    df_updated["custom_norm_term"] = df_updated["term"].parallel_map(lambda x: normalize_gaming(x, "custom"))
    df_updated = append_ipa(df_updated, "eng2ipa", pth.updated_path)
    df_updated = append_ipa(df_updated, "eng2ipa_n", pth.updated_path)
    df_updated = append_ipa(df_updated, "ipa_en", pth.updated_path)
    df_updated = append_ipa(df_updated, "ipa_norm_en", pth.updated_path)
    df_updated = append_ipa(df_updated, "ipa_ms", pth.updated_path)
    df_updated = append_ipa(df_updated, "ipa_norm_ms", pth.updated_path)
    df_updated = append_ipa(df_updated, "en_en_un", pth.updated_path)
    df_updated = append_ipa(df_updated, "ms_en_un", pth.updated_path)
    df_updated = append_ipa(df_updated, "en_en_n", pth.updated_path)
    df_updated = append_ipa(df_updated, "ms_en_n", pth.updated_path)

    # Update unphonemized text with custom phon_mapping
    df_updated["eng2ipa_n"] = df_updated["eng2ipa_n"].parallel_map(phonemize_gaming)

    # Amend id for df_updated i.e. increment id by len(df_gaming)
    df_updated["id"] = df_updated["id"] + len(df_gaming)

    # Print statistics
    ori_len = len(df_gaming)
    add_len = len(additional_terms)
    upd_len = len(updated_terms)
    dup_len = add_len - upd_len
    new_len = ori_len + upd_len

    print(f"\n{'Number of gaming terms in gaming_terms.csv':<60} : {ori_len}")
    print(f"{'Number of additional gaming terms':<60} : {add_len}")
    print(f"{'Number of duplicates removed':<60} : {dup_len}")
    print("Number of additional gaming terms after removing " f"{'duplicates':<11} : {upd_len}")
    print(f"{'Number of gaming terms after adding additional terms':<60} : {new_len}")

    # Append df_updated to df_gaming
    df_combined = pd.concat([df_gaming, df_updated], axis=0)

    # Remove duplicated term
    df_combined = df_combined.drop_duplicates(subset=["term"]).reset_index(drop=True)

    # Save DataFrame
    df_combined.to_csv(pth.combined_path, index=False)

    return df_combined


def gen_text_files(
    data: pd.DataFrame,
    output_dir: str,
) -> Dict[str, List[List[str]]]:
    """Generate text files (i.e. list of lists containing audio_path and phonemes)
    for each IPA columns in DataFrame.
    Save list as csv text file.

    Args:
        data (pd.DataFrame):
            DataFrame containing gaming terms and IPA phonemes.
        output_dir (str):
            String containing directory path to synthesized audio files.

    Returns:
        ipa_dict (Dict[str, List[List[str]]]):
            Dictionary mapping ipa to list of list containing
            audio_path and ipa phonemes.
    """

    # Get list of columns relating to IPA phonemes
    ipa_cols = [col for col in data.columns if "_en" in col or "_ms" in col]

    # Append norm_term to ipa_cols
    total_cols = ipa_cols + ["eng2ipa_n", "norm_term"]
    ipa_dict = {}

    for col in total_cols:
        # Generate filelist based on selected col
        ipa_dict[col] = gen_filelist(data, output_dir, col, col)

    return ipa_dict


def gen_filelist(
    data: pd.DataFrame,
    output_dir: str,
    folder: str,
    text_col: str = "text",
) -> List[List[str]]:
    """Generate filelist for vits batch inferencing; and saved in
    respective folder.

    Args:
        data (pd.DataFrame):
            DataFrame containing text transcript
        output_dir (str):
            String containing directory path to synthesized audio files.
        folder (str):
            Name of folder that contains audio files.
        text_col (str):
            Column containing text transcript.

    Returns:
        (List[List[str]]):
            List of containing audio path and transcript.
    """

    # Extract id column and column containing transcript
    df = data.loc[:, ["id", text_col]]
    records = []

    # Generate list of lists containing audio_path and phonemes
    for row in df.itertuples(index=False, name=None):
        audio_path = f"{output_dir}/{folder}/wav/{row[0]}.wav"
        text = row[1]
        records.append([audio_path, text])

    # Generate text files for input to vits
    text_file_path = f"{output_dir}/{folder}/{folder}.txt"

    # Create folder to hold filelist if not exist
    folder_path = Path(text_file_path).parent

    if not folder_path.is_dir():
        folder_path.mkdir(parents=True, exist_ok=False)

    # Save filelist as text file
    write_to_csv(text_file_path, records)

    return records


def write_to_csv(csv_file: str, records: List[Tuple[str, str]], headers: Optional[List[str]] = None) -> None:
    """Generate csv file in text format.

    Args:
        text_file (str):
            String containing file path to text file used as input to vits inferencing.
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


def detect_phonemes(text: str) -> List[str]:
    """Return list of phonemes detected.

    Args:
        text (str): text string

    Returns:
        (List[str]): List of phoneme symbols detected.
    """

    print(f"Text string : {text}")

    return re.findall(rf"[{LETTERS_IPA}]", text)


def compare_diff(data: pd.DataFrame, pth: DictConfig) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Compare between norm vs unnorm; espeak_en vs espeak_ms; eng2ipa vs ipa_en and
    eng2ipa vs ipa_norm_en

    Args:
        data (pd.DataFrame):
            DataFrame containing gaming terms and ipa phonemes.
        pth (DictConfig):
            Omegaconfig dictionary containing required file paths.

    Returns:
        df_diff (pd.DataFrame):
            DataFrame containing number and percentage of
            different terms between selected columns.
        df_diff_dict (Dict[str, pd.DataFrame]):
            Dictionary containing DataFrames of different columns
            having different phonemes.
        df_same_dict (Dict[str, pd.DataFrame]):
            Dictionary containing DataFrames of different columns
            having different phonemes.
    """

    compare_dict = {
        "eng2ipa_en": ("eng2ipa", "ipa_en"),
        "eng2ipa_en_n": ("eng2ipa_n", "ipa_norm_en"),
        "un_n_term": ("term", "norm_term"),
        "un_n_en": ("ipa_en", "ipa_norm_en"),
        "un_n_ms": ("ipa_ms", "ipa_norm_ms"),
        "en_ms_un": ("ipa_en", "ipa_ms"),
        "en_ms_n": ("ipa_norm_en", "ipa_norm_ms"),
        "en_en(en)_un": ("ipa_en", "en_en_un"),
        "en_ms(en)_un": ("ipa_en", "ms_en_un"),
        "en_en(en)_n": ("ipa_norm_en", "en_en_n"),
        "en_ms(en)_n": ("ipa_norm_en", "ms_en_n"),
    }

    # Initialize dictionary to save same phonemes for the 2 selected phoneme sets.
    df_same_dict = {}

    # Initialize dictionary to save different phonemes for the 2 selected phoneme sets.
    df_diff_dict = {}

    # Initialize dictionary to generate DataFrame containing statistics between
    # phonemes between unnormalized and normalized gaming terms via espeak-en and
    # espeak-ms
    same_dict = {
        col: []
        for col in [
            "col_1",
            "col_2",
            "same_phonemes",
            "same_phonemes_same_terms",
            "same_phonemes_diff_terms",
            "percent_same_phonemes_diff_terms",
        ]
    }

    # Intialize dictionary for generating DataFrame containing statistics for
    # all comparision sets (e.g. eng2ipa_en, etc.) specifically number of
    # terms (num_gaming), number of phonemes that are different (num_diff),
    # number of phonemes that are same (num_same), and percentage difference (percent_diff)
    diff_dict = {
        col: []
        for col in [
            "col_1",
            "col_2",
            "num_gaming",
            "num_diff",
            "num_same",
            "percent_diff",
        ]
    }

    for k, v in compare_dict.items():
        # Extract records that are different between selected columns
        # Update df_diff_dict with filtered DataFrame having different phonemes
        df_diff_dict[k] = data.loc[data[v[0]] != data[v[1]], :].reset_index(drop=True)

        # Extract records that are same between selected columns
        df_same = data.loc[data[v[0]] == data[v[1]], :].reset_index(drop=True)

        # Comparision between normalized and unnormalized phoneme sets
        # for espeak-en and espeak_ms
        if k == "un_n_en" or k == "un_n_ms":
            # Extract out records that have different unnormalized and
            # normalized gaming term but similar phonemes
            df_same_dict[k] = df_same.loc[df_same["term"] != df_same["norm_term"], :].reset_index(drop=True)

            # Compute percentage similarity
            percent_same = round(len(df_same_dict[k]) / len(df_same) * 100, 2)

            # Update same_dict
            same_dict["col_1"].append(v[0])  # First phoneme set
            same_dict["col_2"].append(v[1])  # Second phoneme set to compare with first
            same_dict["same_phonemes"].append(len(df_same))  # Compute number of records havng same phonemes
            same_dict["same_phonemes_same_terms"].append(len(df_same) - len(df_same_dict[k]))
            same_dict["same_phonemes_diff_terms"].append(len(df_same_dict[k]))
            same_dict["percent_same_phonemes_diff_terms"].append(percent_same)

        else:
            # Update df_same_dict with filtered DataFrame having similar phonemes
            df_same_dict[k] = df_same

        # Compute percentage differences
        percent_diff = round(len(df_diff_dict[k]) / len(data) * 100, 2)

        # Save DataFrame for both similar and different phonemes separately
        for cond, df in {"diff": df_diff_dict[k], "same": df_same_dict[k]}.items():
            file_path = Path(pth.gaming_dir).joinpath("dataframe", f"{k}_{cond}.csv")
            df.to_csv(file_path, index=False)

        # Update diff_dict
        diff_dict["col_1"].append(v[0])
        diff_dict["col_2"].append(v[1])
        diff_dict["num_gaming"].append(len(data))
        diff_dict["num_diff"].append(len(df_diff_dict[k]))
        diff_dict["num_same"].append(len(data) - len(df_diff_dict[k]))
        diff_dict["percent_diff"].append(percent_diff)

    # Convert to summary statistics dictionary (i.e. same_dict
    # and diff_dict) to DataFrame
    df_similar = pd.DataFrame(same_dict)
    df_diff = pd.DataFrame(diff_dict)

    # Save DataFrame
    df_similar.to_csv(pth.same_path, index=False)
    df_diff.to_csv(pth.diff_path, index=False)

    return df_diff, df_similar, df_diff_dict, df_same_dict


def display_df(
    data: pd.DataFrame,
    name: Optional[str] = None,
    bkg_color: str = "yellow",
    font_color: str = "black",
) -> pd.DataFrame.style:
    """Highlight required columns based on name of Dataframe.

    Args:
        data (pd.DataFrame]):
            DataFrame of interest.
        name (str):
            Optional string for comparison between columns i.e. "eng2ipa_en",
            "eng2ipa_en_n", "un_n_term", "un_n_en", "un_n_ms", "en_ms_un",
            "en_ms_n", "en_en(en)_un, "en_ms(en)_un", "en_ms(en)_n", or
            "en_ms(en)_n".
        bkg_color (str):
            Highlight column color (Default: "yellow").
        font_color (str):
            Font color (Default: "black")

    Return:
        (pd.DataFrame.style): Pandas styling properties.
    """

    highlight_dict = {
        "eng2ipa_en": ["eng2ipa", "ipa_en", "term"],
        "eng2ipa_en_n": ["eng2ipa_n", "ipa_norm_en", "norm_term"],
        "un_n_term": ["term", "norm_term"],
        "un_n_en": ["ipa_en", "ipa_norm_en", "term", "norm_term"],
        "un_n_ms": ["ipa_ms", "ipa_norm_ms", "term", "norm_term"],
        "en_ms_un": ["ipa_en", "ipa_ms", "term"],
        "en_ms_n": ["ipa_norm_en", "ipa_norm_ms", "norm_term"],
        "en_en(en)_un": ["ipa_en", "en_en_un", "term"],
        "en_ms(en)_un": ["ipa_en", "ms_en_un", "term"],
        "en_en(en)_n": ["ipa_norm_en", "en_en_n", "norm_term"],
        "en_ms(en)_n": ["ipa_norm_en", "ms_en_n", "norm_term"],
    }

    if name:
        print(f"Number of similar records : {len(data)}\n")
        subset = highlight_dict[name]

    else:
        # If no comparision set, then extract required row i.e. (ipa_en, ipa_ms)
        # and (ipa_norm_en, ipa_norm_ms) for the purpose of displaying summary
        # statistics DataFrame (i.e. )
        df_row = data.loc[
            data["col_1"].str.contains("ipa_en|ipa_ms") & data["col_2"].str.contains("ipa_norm_"),
            :,
        ]

        # Pass subset DataFrame index and column to pd.IndexSlice
        subset = pd.IndexSlice[df_row.index, df_row.columns]

    return data.style.set_properties(
        subset=subset,
        **{"background-color": bkg_color, "color": font_color},
    )


def modify_path(
    text_file_path: str,
    dest_audio_dir: str,
    new_file_path: Optional[str] = None,
) -> None:
    """Modify the file path of the audio output wav to be
    relative to dest_dir in text file.

    Args:
        text_file_path (str):
            String containing file path to text file.
        dest_dir (str):
            String containing directory path to synthesized audio files.
        new_file_path (Optional[str]):
            String containing new file path to save modified text.
    """

    records = []

    # Read original text file and modify audio_path
    with open(text_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            # row[0] = audio_path; row[1] = text
            text = row[1].strip()

            # Existing audio directory indicated in audio path
            existing_audio_dir = Path(row[0]).parent.as_posix()

            # Amend audio output directory to point to dest_audio_dir
            output_path = row[0].replace(
                existing_audio_dir,
                dest_audio_dir,
            )

            records.append((output_path, text))

    # Update text file
    if new_file_path:
        write_to_csv(new_file_path, records)
    else:
        write_to_csv(text_file_path, records)

    return None


def normalize_oov(text_file_path: str, mapping: Dict[str, str] = {"+": " dan "}) -> None:
    """Normalize oov words in vits filelist.

    Args:
        text_file_path (str):
            String containing file path to text file.
        mapping(Dict[str, str]):
            Dictionary containing oov words and its replacement.

    Returns:
        None
    """

    records = []

    # Read original text file and normalize oov words
    with open(text_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            # row[0] = audio_path; row[1] = text
            audio_path = row[0].strip()
            text = row[1]

            # Normalize oov words i.e. '+'
            for oov, replaced in mapping.items():
                text = re.sub(re.escape(oov), replaced, text)

            records.append((audio_path, remove_extra_whitespace(text)))

    # Update vits filelist
    write_to_csv(text_file_path, records)

    return None


def gen_wav_file(text: str, file_path: str, model="mesolitica/VITS-osman", sampling_rate: int = 22050) -> None:
    """Generate malay audio using malaya_speech pre-trained vits model.

    Args:
        text (str):
            text string
        file_path (str):
            String containing file path to synthesized audio.
        model (str, optional):
            TTS pre-trained model (Default: "mesolitica/VITS-osman").
        sampling_rate (int):
            Sampline rate in hertz (Default: 22050).

    Returns:
        None
    """

    # Load TTS model once
    global VITS_MODEL
    if VITS_MODEL is None:
        VITS_MODEL = malaya_speech.tts.vits(model="mesolitica/VITS-osman")

    # Perform waveform prediction
    pred_waveform = VITS_MODEL.predict(text)["y"]

    # Create Audio object
    audios = ipd.Audio(pred_waveform, rate=sampling_rate)

    # Save audio as WAV file
    with open(file_path, "wb") as wav_file:
        wav_file.write(audios.data)

    del audios
    del pred_waveform
    gc.collect()

    return None


def gen_audio(data: pd.DataFrame, col: str, audio_dir: str) -> None:
    """Generate Audio for selected column in DataFrame; and save
    synthesized audio in specified audio_dir.

    Args:
        data (pd.DataFrame):
            DataFrame containing gaming terms and its phonemes.
        col (str):
            Selected column in DataFrame typically either "term" or "norm_term".
        audio_dir (str):
            String containing directory path to audio folder.
    """

    # Extract relevant column
    df = data.loc[:, ["id", col]]

    # Ensure audio directory is absolute path format
    audio_dir = Path(audio_dir).expanduser().resolve()

    # Create audio directory if not exist
    if not audio_dir.is_dir():
        audio_dir.mkdir(parents=True, exist_ok=False)

    # Generate column containing file paths for each generated wav file
    df["audio_path"] = df["id"].map(lambda x: audio_dir.joinpath(f"{x}.wav").as_posix())

    df["temp"] = df.progress_apply(lambda row: gen_wav_file(row[col], row["audio_path"]), axis=1)

    return None


def append_gaming(
    data: pd.DataFrame,
    text_file_path: str,
    dest_dir: str,
    required_cols: Tuple[str] = ("id", "norm_malaya"),
) -> None:
    """Append gaming terms to text_file for vits training.

    Args:
        data (pd.DataFrame):
            DataFrame containing normalized gaming terms.
        required_cols (List[str]):
            Tuple of required columns (Default: ("id", "norm_malaya")).
        dest_dir (str):
            String containing directory path to synthesized audio files.
        text_file_path (str):
            String containing file path to vits file lists.

    Returns:
        None
    """

    # Extract required columns from DataFrame
    df = data.loc[:, required_cols]

    with open(text_file_path, "a", newline="") as f:
        # Set csv writer delimiter to be "|"
        writer = csv.writer(f, delimiter="|")

        # Append gaming terms to vits file list
        for row in df.itertuples(index=False):
            audio_path = f"{dest_dir}/{row[0]}.wav"
            writer.writerow([audio_path.strip(), row[1].strip()])

    return None


def check_filelist(text_file_path: str) -> pd.DataFrame:
    """Convert text to ascii and check for out of vocab symbols.

    Args:
        text_file_path (str):
            String containing file path to vits filelist.

    Returns:
        df (pd.DataFrame):
            DataFrame containing oov terms
    """

    oov_dict = {
        "audio_path": [],
        "text": [],
        "text_ascii": [],
        "oov": [],
    }

    # Read file
    with open(text_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")

        for row in reader:
            # Convert to ascii
            text = row[1].strip()
            text_ascii = unidecode(text)

            # Update dictionary
            oov_dict["audio_path"].append(row[0].strip())
            oov_dict["text"].append(text)
            oov_dict["text_ascii"].append(text_ascii)

            # Check for oov symbols
            oov_dict["oov"].append([symb for symb in text_ascii if symb not in symbols])

    # Convert Dictionary to DataFrame
    df = pd.DataFrame(oov_dict)

    # Insert id column
    df.insert(0, "id", df["audio_path"].map(lambda x: Path(x).stem))

    return df


def gen_audio_or_eval(gaming_dir: str, yaml_path: str, combined_path: str) -> None:
    """Generate audio files for phonemes set specified in `col_list` or
    Perform evaluation of audio files for phonemes.

    Args:
        gaming_dir (str):
            Absolute path to directory designated to store
            gaming related audio files.
        yaml_path (str):
            Absolute path to `batch_inferencing.yaml` or `model_objective_eval.yaml'
        combined_path:
            Absolute path to `combined_gaming_terms.yaml`.

    Returns:
        None
    """

    # Load OmegaConf dictionary from `batch_inferencing.yaml`
    cfg = OmegaConf.load(yaml_path)
    yaml_path = Path(yaml_path)

    # Perform inference or model evaluation based on yaml file path
    mode = (
        ["inferencing", vits_batch_infer] if yaml_path.name == "batch_inferencing.yaml" else ["evaluation", eval_phon]
    )

    # Get List of columns containing IPA phonemes from `combined_gaming_terms.csv`
    combined = pd.read_csv(combined_path)
    ipa_cols = [col for col in combined.columns if "_n" in col or "_en" in col or "_ms" in col]

    logging.info("Performing %s...", mode[0])

    for col in ipa_cols:
        # Get updated paths for batch inferencing
        syn_path = Path(gaming_dir).joinpath(col)
        text_file = syn_path.joinpath(f"{col}.txt")
        output_csv_filepath = syn_path.joinpath(f"output_{col}.csv")

        # Get updated paths for model evaluation
        syn_wav_path = syn_path.joinpath("wav")
        syn_mel_dir = syn_path.joinpath("mel")
        syn_mfcc_dir = syn_path.joinpath("mfcc")
        mfcc_path = syn_path.joinpath("dataframe", f"output_{col}_mfcc.csv")
        acr_path = syn_path.joinpath("dataframe", f"output_{col}_acr.csv")
        mcd_path = syn_path.joinpath("dataframe", f"output_{col}_mcd.csv")
        asr_path = syn_path.joinpath("dataframe", f"output_{col}_asr.csv")
        wer_cer_path = syn_path.joinpath("dataframe", f"output_{col}_wer_cer.csv")
        complete_path = syn_path.joinpath(f"output_{col}_complete.csv")

        # Ensure text manifest (i.e. <col>.txt) is present for inferencing and;
        # output csv file (i.e. output_<col>.csv) is present for model evaluation
        required_input_path = text_file if yaml_path.name == "batch_inferencing.yaml" else output_csv_filepath

        if required_input_path.is_file():
            logging.info("required_input_path (%s) is present", required_input_path.name)

            with open_dict(cfg):
                # Update cfg for batch inferencing
                if yaml_path.name == "batch_inferencing.yaml":
                    cfg.inference.text_file = text_file.as_posix()
                    cfg.inference.output_csv_filepath = output_csv_filepath.as_posix()

                    logging.info("Updated cfg.inference.text_file: %s", cfg.inference.text_file)
                    logging.info("Updated cfg.inference.output_csv_filepath: %s", cfg.inference.output_csv_filepath)

                # Update cfg for model evaluation
                else:
                    cfg.path.syn_path = syn_path.as_posix()
                    cfg.path.syn_wav_path = syn_wav_path.as_posix()
                    cfg.path.syn_mel_dir = syn_mel_dir.as_posix()
                    cfg.path.syn_mfcc_dir = syn_mfcc_dir.as_posix()
                    cfg.path.test_csv_path = output_csv_filepath.as_posix()
                    cfg.path.test_mfcc_path = mfcc_path.as_posix()
                    cfg.path.test_acr_path = acr_path.as_posix()
                    cfg.path.test_mcd_path = mcd_path.as_posix()
                    cfg.path.test_asr_path = asr_path.as_posix()
                    cfg.path.test_wer_cer_path = wer_cer_path.as_posix()
                    cfg.path.test_complete_path = complete_path.as_posix()

                    logging.info("Updated cfg.path.syn_path: %s", cfg.path.syn_path)
                    logging.info("Updated cfg.path.syn_wav_path: %s", cfg.path.syn_wav_path)
                    logging.info("Updated cfg.path.syn_mel_dir: %s", cfg.path.syn_mel_dir)
                    logging.info("Updated cfg.path.syn_mfcc_dir: %s", cfg.path.syn_mfcc_dir)
                    logging.info("Updated cfg.path.test_csv_path: %s", cfg.path.test_csv_path)
                    logging.info("Updated cfg.path.test_mfcc_path: %s", cfg.path.test_mfcc_path)
                    logging.info("Updated cfg.path.test_acr_path: %s", cfg.path.test_acr_path)
                    logging.info("Updated cfg.path.test_mcd_path: %s", cfg.path.test_mcd_path)
                    logging.info("Updated cfg.path.test_asr_path: %s", cfg.path.test_asr_path)
                    logging.info("Updated cfg.path.test_wer_cer_path: %s", cfg.path.test_wer_cer_path)
                    logging.info("Updated cfg.path.test_complete_path: %s", cfg.path.test_complete_path)

            logging.info("Start %s for %s...", mode[0], col)

            # Perform inferencing or model evaluation depending on yaml file input
            # mode[1] = vits_batch_infer or eval_phon function
            mode[1](cfg)

            logging.info("Successfully completed %s for %s", mode[0], col)

        else:
            logging.info("required_input_path (%s) is absent!", required_input_path.name)


# if __name__ == "__main__":
#     # initalize_pandarallel(True, 4)

#     cfg = OmegaConf.load("/home/ckabundant/Documents/tts-melayu/conf/base/gaming.yaml")
#     df_gpt = pd.read_csv(cfg.paths.gpt_csv_path)

#     gen_gpt_audio(df_gpt, "norm_gpt", cfg.paths.gpt_dir, start_index=1119, batch_size=100)
