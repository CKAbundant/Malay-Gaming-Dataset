""" from https://github.com/keithito/tacotron """
# from text import cleaners
# from text.symbols import symbols
import os
import re
import tempfile
from typing import Dict, List, Optional

import hydra
import tiktoken
import torch
from omegaconf import DictConfig

# Incorporated from FB MMS TTS Inference code
from .. import commons
from . import cleaners

# from ..text_processing import TextMapper


# PAD = "_"
# PUNCTUATION = ';:,.!?¡¿—…"«»“” '
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθ"
# LETTERS_IPA = LETTERS_IPA + "œɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# # Use this and comment out the line for 'symbols' below if you intend to use
# # Facebook research MMS-TTS pretrained zlm models
# # fb_symbols = "yg feto5j_3k–ia0n6duc'hq-pmwr4slbz"
# # symbols = list(fb_symbols)

# # Export all symbols:
# symbols = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(LETTERS_IPA)

# # Special symbol ids
# SPACE_ID = symbols.index(" ")

# # Mappings from symbol to numeric ID and vice versa:
# _symbol_to_id = {s: i for i, s in enumerate(symbols)}
# # _id_to_symbol = {i: s for i, s in enumerate(symbols)}
# _id_to_symbol = dict(enumerate(symbols))
_symbol_to_id = None
_id_to_symbol = None


def _clean_text(text_string: str, cleaner_names: List[str], dp_model_filepath: Optional[str] = None) -> str:
    """
    Cleans a given text string using specified cleaner functions.

    This function iterates through a list of cleaner names, applying each
    cleaner to the text string.
    If a cleaner name includes 'deep_phonemizer', and a model file path is
    provided and exists, the
    cleaner is called with the model file path. Otherwise, each cleaner is
    applied normally.

    Args:
        text_string (str): The text string to be cleaned.
        cleaner_names (list of str): A list of names of the cleaners to be
        applied.
            Each name must correspond to a cleaner function in the
            cleaners.
        dp_model_filepath (str, optional): The file path to the model used by
        the 'deep_phonemizer' cleaner.
            Defaults to None.

    Returns:
        str: The cleaned text string.

    Raises:
        ValueError: If any cleaner name in cleaner_names does not correspond
        to a known cleaner function.
    """
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise ValueError(f"Unknown cleaner: {name}")

        # Check if the cleaner name contains 'deep_phonemizer'
        if re.search("deep_phonemizer", name):
            # Check if the model file path exists
            if dp_model_filepath is not None and os.path.exists(dp_model_filepath):
                # Call the cleaner with the specified model file path
                text_string = cleaner(text_string, dp_model_filepath)
            else:
                text_string = cleaner(text_string)
        else:
            # Call the cleaner normally
            text_string = cleaner(text_string)

    return text_string


class TextMapperTrain(object):
    """A class for mapping text to sequences of symbol IDs and vice versa,
    along with other text processing functionalities.

    This class is designed to convert text into sequences of symbol IDs
    based on a vocabulary file,
    and apply various text processing methods, such as cleaning and
    romanization.

    Attributes:
        symbols (list of str): A list of symbols from the vocabulary file.
        SPACE_ID (int): The index of the space character in the symbols list.
        _symbol_to_id (dict): A dictionary mapping symbols to their
        corresponding IDs.
        _id_to_symbol (dict): A dictionary mapping IDs to their corresponding
        symbols.

    Methods:
        text_to_sequence(text, hps): Converts a string of text to a sequence
        of symbol IDs.
        uromanize(text, uroman_pl): Romanizes a given text using the Uroman
        tool.
        get_text(text, hps): Converts a string of text to a torch.LongTensor
        of symbol IDs.
        filter_oov(text): Filters out characters in the text that are not in
        the vocabulary.
    """

    def __init__(self, vocab_file: str):
        """
        Initializes the TextMapperTrain with a vocabulary file.

        Args:
            vocab_file (str): The path to the vocabulary file.
        """
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        # print(f"symbols: \n {self.symbols}")
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    # def text_to_sequence(self, text, cleaner_names):
    def text_to_sequence(self, text: str, hps: DictConfig) -> List[int]:
        """
        Converts a string of text to a sequence of IDs corresponding to the
        symbols in the text.

        Args:
            text (str): The string to convert to a sequence.
            hps (DictConfig): A configuration object containing
            hyperparameters and settings.

        Returns:
            List[int]: A list of integers corresponding to the symbols
            in the text.
        """
        sequence = []
        cleaned_text = text.strip()
        ################################################################
        # Not in original FB MMS TTS code, adapted from original VITS
        # Adapted code to set deep_phonemizer model file as configurable
        text_cleaners_type = hps.vits_train.data.text_cleaners
        if re.search("deep_phonemizer", text_cleaners_type[0]) and hps.deep_phonemizer.dp_model_filepath is not None:
            dp_model_filepath = hydra.utils.to_absolute_path(hps.deep_phonemizer.dp_model_filepath)
            cleaned_text = _clean_text(cleaned_text, text_cleaners_type, dp_model_filepath)
        else:
            cleaned_text = _clean_text(cleaned_text, text_cleaners_type)
        # clean_text = _clean_text(clean_text, cleaner_names)
        ## #############################################################
        for symbol in cleaned_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
        return sequence

    def uromanize(self, text: str, uroman_pl: str) -> str:
        """
        Romanizes the given text using the Uroman tool.

        Args:
            text (str): The text to be romanized.
            uroman_pl (str): The path to the Uroman Perl script.

        Returns:
            str: The romanized text.
        """
        iso = "xxx"
        with tempfile.NamedTemporaryFile() as tf, tempfile.NamedTemporaryFile() as tf2:
            with open(tf.name, "w") as f:
                f.write("\n".join([text]))
            cmd = "perl " + uroman_pl
            cmd += f" -l {iso} "
            cmd += f" < {tf.name} > {tf2.name}"
            os.system(cmd)
            outtexts = []
            with open(tf2.name) as f:
                for line in f:
                    line = re.sub(r"\s+", " ", line).strip()
                    outtexts.append(line)
            outtext = outtexts[0]
        return outtext

    def get_text(self, text: str, hps: DictConfig) -> torch.LongTensor:
        """
        Converts a string of text to a torch.LongTensor of symbol IDs.

        Args:
            text (str): The text to convert.
            hps: A configuration object containing hyperparameters.

        Returns:
            torch.LongTensor: A tensor of symbol IDs representing the text.
        """
        # text_norm = self.text_to_sequence(text, hps.data.text_cleaners)
        # Adapted code to set deep_phonemizer model file as configurable
        text_norm = self.text_to_sequence(text, hps)
        # if hps.data.add_blank:
        if hps.vits_train.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def filter_oov(self, text: str) -> str:
        """
        Filters out characters in the text that are not in the vocabulary.

        Args:
            text (str): The text to filter.

        Returns:
            str: The filtered text, containing only characters present in
            the vocabulary.
        """
        val_chars = self._symbol_to_id
        txt_filt = "".join(list(filter(lambda x: x in val_chars, text)))
        print(f"text after filtering OOV: {txt_filt}")
        return txt_filt


def init_mapping(text_mapper: TextMapperTrain) -> None:
    """Generate symbol to id and id to symbol mapping dictionary
    based on given vocab text file path.

    Args:
        text_mapper (TextMapperTrain):
            Instance of TextMapperTrain Object pertaining to selected vocab text file.

    Returns:
        None
    """

    global _symbol_to_id
    global _id_to_symbol

    if _symbol_to_id is None:
        _symbol_to_id = text_mapper._symbol_to_id

    if _id_to_symbol is None:
        _id_to_symbol = text_mapper._id_to_symbol

    return None


# HYL: g2p, then tokenize phoneme
def text_to_sequence(text: str, text_mapper: TextMapperTrain, cleaner_names: List[str]) -> List[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
        text (str):
            String to convert to a sequence.
        text_mapper (TextMapperTrain):
            Instance of TextMapperTrain Object pertaining to selected vocab text file.
        cleaner_names (List[str]):
            Names of the cleaner functions to run the text through
    Returns:
        (List[int]): List of integers corresponding to the symbols in the text
    """

    # Intialize _symbol_to_id if None
    init_mapping(text_mapper)

    # Clean and phonemized text based on list of cleaners in cleaner_names
    cleaned_text = clean_text(text, cleaner_names)

    return [_symbol_to_id[symbol] for symbol in cleaned_text]


# tokenize phoneme
def cleaned_text_to_sequence(cleaned_text, text_mapper: TextMapperTrain) -> List[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    Args:
        cleaned_text (str):
            cleaned text/phonemized string to convert to a integer sequence.
        text_mapper (TextMapperTrain):
            Instance of TextMapperTrain Object pertaining to selected vocab text file.
    Returns:
        (List[int]):
            List of integers corresponding to the symbols in the text.
    """

    # Intialize _symbol_to_id if None
    init_mapping(text_mapper)

    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]

    return sequence


# tokenize subword
def cleaned_text_to_sequence_subword(cleaned_text, tokenizer=None):
    """Converts a string of text to a sequence using Tokenizer.

    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        sequence = tokenizer.encode(cleaned_text)
    else:
        sequence = tokenizer.encode(cleaned_text).ids

    return sequence


# decode token back to phoneme
def sequence_to_text(sequence, text_mapper):
    """
    Convert a sequence of IDs back to a string.

    This function translates a list of integer IDs into the
    corresponding symbols (e.g., characters) and then
    concatenates them to produce a string.

    Args:
        sequence (List[int]): A list of integer IDs to be
        converted into their corresponding symbols.

    Returns:
        str: The resulting string that represents the sequence of IDs.
    """

    # Intialize _id_to_symbol if None
    init_mapping(text_mapper)

    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s

    return result


# load g2p (or potentially normalizer) and apply
def clean_text(text, cleaner_names):
    """
    Clean the input text using the specified cleaner functions.

    This function applies a series of cleaning operations
    to a given input text. The specific cleaning operations
    to apply are determined by the `cleaner_names` argument.
    Each cleaner name corresponds to a cleaning function
    defined in the `cleaners` module.

    Args:
        text (str): The input string to be cleaned.
        cleaner_names (List[str]): A list of string names
        representing cleaning functions to apply to the text.

    Returns:
        str: The cleaned text.

    Raises:
        ValueError: If any of the cleaner names provided in
        `cleaner_names` is not found in the `cleaners` module.
    """
    for name in cleaner_names:
        # cleaner = getattr(cleaners, name)
        cleaner = getattr(cleaners, name)
        if not cleaner:
            # pylint: disable=broad-exception-raised
            raise Exception(f"Unknown cleaner: {name}")
            # raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
