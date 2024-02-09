"""
This script is adapted from the text processing module of the Tacotron model,
designed to define and handle the set of symbols used in text input for
speech synthesis.
It includes functions for text cleaning, phoneme-to-symbol conversion,
and other preprocessing steps necessary for preparing text for the Tacotron
model.

The script defines a set of symbols including punctuation, letters, and
International
Phonetic Alphabet (IPA) symbols. It provides functionality to clean text
using specified
cleaners, convert text to sequences of symbol IDs, intersperse symbols with
specified items,
and process text for use with a speech synthesis model.

Classes:
    TextMapper: Maps text to sequences of symbol IDs and vice versa, with
    additional text processing functionalities.

Functions:
    _clean_text(text_string: str, cleaner_names: List[str],
    dp_model_filepath: Optional[str]) -> str:
        Cleans a text string using specified cleaner functions.

    text_to_sequence(text_string: str, hps: DictConfig) -> List[int]:
        Converts text to a sequence of symbol IDs.

    intersperse(lst: List[Any], item: Any) -> List[Any]:
        Inserts an item between each element of a list.

    get_text(text_string: str, hps: DictConfig) -> torch.LongTensor:
        Processes text string and returns a tensor of symbol IDs.

    get_text_subword(text_string: str, hps) -> torch.LongTensor:
        Processes text string for subword models and returns a tensor of
        symbol IDs.

    preprocess_char(text: str, lang: Optional[str]) -> str:
        Applies language-specific preprocessing to characters in a text string.

    preprocess_text(txt: str, text_mapper, hps: DictConfig,
    uroman_dir: Optional[str], lang: Optional[str]) -> str:
        Preprocesses a text string by applying character preprocessing
        and optional romanization.

Typical usage example:
    text_mapper = TextMapper(vocab_file)
    processed_text = preprocess_text("Hello world", text_mapper, hps_config)
"""

# Incorporated from FB MMS TTS Inference code and tested on FASTAPI inference
import os
import re
import subprocess
import tempfile
from typing import Any, List, Optional

import hydra
import torch
from omegaconf import DictConfig

# Incorporated from FB MMS TTS Inference code
from . import commons
from . import text as text_module

VERBOSE = False

PAD_T = "_"
PUNCTUATION_T = ';:,.!?¡¿—…"«»“” '
LETTERS_T = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# rewritten _letters_ipa to be concatenation of various chunks of ipa symbols
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺ\
# ɾɻʀʁɽʂ\ʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂ"
LETTERS_IPA = LETTERS_IPA + "ʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Use this and comment out the line for 'symbols' below if you intend to use
# Facebook research MMS-TTS pretrained zlm models
# fb_symbols = "yg feto5j_3k–ia0n6duc'hq-pmwr4slbz"
# symbols = list(fb_symbols)

# Export all symbols:
symbols = [PAD_T] + list(PUNCTUATION_T) + list(LETTERS_T) + list(LETTERS_IPA)

# Special symbol ids
SPACE_ID = symbols.index(" ")
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
# _id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Adapted code to set deep_phonemizer model file as configurable
# def _clean_text(text, cleaner_names):


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
            text_module.cleaners.
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
        cleaner = getattr(text_module.cleaners, name)
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


# G2P, then tokenize
def text_to_sequence(text_string: str, hps: DictConfig) -> List[int]:
    """
    Convert the given text string to a sequence of symbol IDs.

    Args:
        text_string (str): The input text.
        hps (DictConfig): Omegaconf Dictionary containing
            data.text_cleaners: List of cleaner names to use for text cleaning.
            data.cleaned_text: Whether text has been phonemized

    Returns:
        (List): A list of symbol IDs representing the text.
    """

    if hps.data.cleaned_text:  # text_string is already cleaned and phonemized
        clean_text = text_string

    else:
        clean_text = _clean_text(text_string, hps.data.text_cleaners)

    # Convert symbol to integer ids.
    return [_symbol_to_id[symbol] for symbol in clean_text]


def intersperse(lst: List[Any], item: Any) -> List[Any]:
    """
    Insert the given item between each element of the list.

    Args:
        lst (list): The list to be modified.
        item (any): The item to insert between each element of lst.

    Returns:
        list: A new list with the item interspersed.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def get_text(text_string: str, hps: DictConfig) -> torch.LongTensor:
    """
    Process the input text string and return a tensor of symbol IDs.

    Args:
        text_string (str): The input text.
        hps (Hyperparams): Hyperparameters containing text cleaners and
        other configurations.

    Returns:
        (torch.LongTensor): Tensor of symbol IDs representing text string.
    """

    # Normalize and phonemize text
    text_norm = text_to_sequence(text_string, hps)

    # Intersperse symbol ids with white space if hps.data.add_bank is True
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)

    # Convert phonemized text to torch LongTensor
    text_norm = torch.LongTensor(text_norm)

    return text_norm


# tokenization, add_blank, convert to torch tensor
def get_text_subword(text_string: str, hps) -> torch.LongTensor:
    """
    Process the input text string and return a tensor of symbol IDs.

    Args:
        text_string (str): The input text.
        hps (Hyperparams): Hyperparameters containing text cleaners
        and other configurations.

    Returns:
        torch.LongTensor: Tensor of symbol IDs representing the text.
    """
    if VERBOSE:
        print("text used for subword is\n", text_string)
    text_norm = text_module.cleaned_text_to_sequence_subword(text_string)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


# Incorporated preprocess_char, preprocess_text functions
# and TextMapper class from FB MMS TTS Inference code


def preprocess_char(text: str, lang: Optional[str] = None) -> str:
    """
    Applies language-specific preprocessing to characters in a text string.

    This function provides special treatment for characters in some languages.
    For example, in Romanian ('ron'), it replaces the character 'ț' with 'ţ'.
    If no specific language treatment is defined, the text is returned as is.

    Args:
        text (str): The text string to be processed.
        lang (str, optional): A code representing the language for which
            special character processing is required. Defaults to None.

    Returns:
        str: The preprocessed text string.

    Note:
        Currently, this function only includes special processing for
        Romanian ('ron').
        Additional languages and character treatments can be added as needed.
    """
    print(lang)
    if lang == "ron":
        text = text.replace("ț", "ţ")
    return text


class TextMapper(object):
    """
    A class for mapping text to sequences of symbol IDs and vice versa,
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
        Initializes the TextMapper with a vocabulary file.

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
        clean_text = text.strip()
        ################################################################
        # Not in original FB MMS TTS code, adapted from original VITS
        # Adapted code to set deep_phonemizer model file as configurable
        text_cleaners_type = hps.vits_train.data.text_cleaners
        if re.search("deep_phonemizer", text_cleaners_type[0]) and hps.deep_phonemizer.dp_model_filepath is not None:
            dp_model_filepath = hydra.utils.to_absolute_path(hps.deep_phonemizer.dp_model_filepath)
            clean_text = _clean_text(clean_text, text_cleaners_type, dp_model_filepath)
        else:
            clean_text = _clean_text(clean_text, text_cleaners_type)
        # clean_text = _clean_text(clean_text, cleaner_names)
        ## #############################################################
        for symbol in clean_text:
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


def preprocess_text(
    txt: str, text_mapper, hps: DictConfig, uroman_dir: Optional[str] = None, lang: Optional[str] = None
) -> str:
    """
    Preprocesses a given text string by applying character preprocessing,
    optional romanization,
    and filtering out characters not present in the vocabulary.

    The function first applies language-specific character preprocessing.
    If the text source is indicated as Uroman ('uroman' extension in training
    files), it clones the Uroman tool and applies romanization to the text.
    The text is then converted to lowercase and filtered to remove characters
    not present in the vocabulary.

    Args:
        txt (str): The text string to be preprocessed.
        text_mapper: An instance of TextMapper for text processing and mapping.
        hps: A configuration object containing hyperparameters and settings.
        uroman_dir (Optional[str]): The directory where the Uroman tool is
        located.
            If None, Uroman is cloned from its repository. Defaults to None.
        lang (Optional[str]): A code representing the language for special
        character processing.
            Defaults to None.

    Returns:
        str: The preprocessed text string.

    Note:
        Uroman romanization is applied only if the training files have a
        'uroman' extension.
    """
    txt = preprocess_char(txt, lang=lang)
    is_uroman = hps.data.training_files.split(".")[-1] == "uroman"
    if is_uroman:
        with tempfile.TemporaryDirectory() as tmp_dir:
            if uroman_dir is None:
                cmd = f"git clone git@github.com:isi-nlp/uroman.git {tmp_dir}"
                print(cmd)
                subprocess.check_output(cmd, shell=True)
                uroman_dir = tmp_dir
            uroman_pl = os.path.join(uroman_dir, "bin", "uroman.pl")
            print("uromanize")
            txt = text_mapper.uromanize(txt, uroman_pl)
            print(f"uroman text: {txt}")
    txt = txt.lower()
    txt = text_mapper.filter_oov(txt)
    return txt
