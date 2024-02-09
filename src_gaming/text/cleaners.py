"""Adapted from https://github.com/keithito/tacotron

This script is part of the text processing module for speech synthesis models
like Tacotron.

It defines various 'cleaners' - functions that preprocess text data at both
training and evaluation time.

The script includes several functions designed to handle different types of
text processing tasks.
These include expanding abbreviations, converting text to lowercase,
collapsing whitespace,
converting non-ASCII characters to ASCII, and more complex operations like
phonemization of text.

The script also provides specific cleaning pipelines for different languages
and use cases.
For English text, 'english_cleaners' and 'english_cleaners2' are available.
For non-English text that can
be transliterated to ASCII, 'transliteration_cleaners' is suitable. Basic
cleaning without transliteration
can be done using 'basic_cleaners'. Additionally, specialized cleaners for
Malay text ('malay_cleaners'
and 'malay_cleaners2') are provided, along with a deep phonemizer for Malay.

Functions:
    expand_abbreviations(text: str) -> str:
        Expands abbreviations in the given text.

    lowercase(text: str) -> str:
        Converts the given text to lowercase.

    collapse_whitespace(text: str) -> str:
        Collapses multiple whitespaces into a single space.

    convert_to_ascii(text: str) -> str:
        Converts non-ASCII characters in the text to their ASCII equivalents.

    basic_cleaners(text: str) -> str:
        Performs basic cleaning on the text without transliteration.

    transliteration_cleaners(text: str) -> str:
        Cleans and transliterates non-English text to ASCII.

    english_cleaners(text: str) -> str:
        Cleans English text and phonemizes it.

    english_cleaners2(text: str) -> str:
        Cleans English text, phonemizes it, includes punctuation and stress.

    malay_cleaners(text: str) -> str:
        Cleans Malay text and phonemizes it.

    malay_cleaners2(text: str) -> str:
        Cleans Malay text, phonemizes it, and includes punctuation and stress.

    malay_deep_phonemizer(text: str, dp_model_filepath: str) -> str:
        Cleans Malay text and phonemizes it using a deep phonemizer model.

These functions enable flexible text preprocessing tailored to the
requirements of various languages
and text-to-speech synthesis models.
"""
import re

from dp.phonemizer import Phonemizer
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text: str) -> str:
    """Expand abbreviations to full word.

    Args:
        text (str):
            Text transcript containing abbreviations.

    Returns:
        text (str):
            Text transcripts with expanded abbreviations.
    """

    # Expand abbreviations based on List of tuples of
    # regex pattern and corresponding full word
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)

    return text


def lowercase(text: str) -> str:
    """Change to text string to lowercase.

    Args:
        text (str):
            Text transcript.

    Returns:
        (str):
            Lowercase text transcript.
    """

    return text.lower()


def collapse_whitespace(text: str) -> str:
    """Remove extra whitespaces in text transcript.

    Args:
        text (str):
            Text transcript.

    Returns:
        (str):
            Text transcript with extra whitespaces removed.
    """

    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text: str) -> str:
    """Convert non-ascii character in text transcript to ascii character.

    Args:
        text (str):
            Text transcript.

    Returns:
        (str):
            Text transcript with only ascii characters.
    """

    return unidecode(text)


def basic_cleaners(text: str) -> str:
    """Basic pipeline that lowercases and collapses whitespace
    without transliteration.

    Args:
        text (str):
            Text transcript.

    Returns:
        text (str):
            Lowercase text transcript with extra whitespaces removed.

    """

    text = lowercase(text)
    text = collapse_whitespace(text)

    return text


def transliteration_cleaners(text: str) -> str:
    """Pipeline for non-English text that transliterates to ASCII.

    Args:
        text (str):
            Text transcript.

    Returns:
        text (str):
            Lowercase text transcript with extra whitespaces removed
            and only ascii characters present.
    """
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)

    # replace x with ks, since x is not used as a token in FB MMS
    text = text.replace("x", "ks")
    # replace v with f, since v is not used as a token in FB MMS
    text = text.replace("v", "f")

    return text


def english_cleaners(text: str) -> str:
    """Pipeline for English text, including abbreviation expansion.

    Args:
        text (str):
            English text transcript.

    Returns:
        phonemes (str):
            Cleaned phonemized lowercased English text transcript
            with only ascii characters.
    """

    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)

    return phonemes


def english_cleaners2(text: str) -> str:
    """Pipeline for English text, including abbreviation expansion
    + punctuation + stress.

    Args:
        text (str):
            English text transcript.

    Returns:
        phonemes (str):
            Cleaned phonemized lowercased text transcript with only
            ascii characters and stressed intonation.
    """

    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
    )

    phonemes = collapse_whitespace(phonemes)

    return phonemes


# Added in malay_cleaners and malay_cleaners2
# for Osman datasets


def malay_cleaners(text: str) -> str:
    """Pipeline for Malay text, including abbreviation expansion.

    Args:
        text (str):
            Text string to be phonemized.

    Returns:
        phonemes (str):
            Phonemized text string.
    """

    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Remove language flags for phonemes
    phonemes = phonemize(
        text,
        language="ms",
        backend="espeak",
        strip=True,
        language_switch="remove-flags",
    )

    phonemes = collapse_whitespace(phonemes)

    return phonemes


def malay_cleaners2(text: str) -> str:
    """Pipeline for Malay text, including abbreviation expansion.
    + punctuation + stress

    Args:
        text (str):
            Text string to be phonemized.

    Returns:
        phonemes (str):
            Phonemized text string.
    """

    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Preserve punctuations and with intonations
    phonemes = phonemize(
        text,
        language="ms",
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )

    phonemes = collapse_whitespace(phonemes)

    return phonemes


def malay_deep_phonemizer(text: str, dp_model_filepath: str = "vits/text/dp_ms_model.pt") -> str:
    """Pipeline for Malay text, via pretrained deep_phonemizer model.

    Args:
        text (str):
            Text string to be phonemized.

    Returns:
        phonemes (str):
            Phonemized text string.
    """

    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)

    # Preserve punctuations and with intonations
    # dp_model_filepath = "vits/text/dp_ms_model.pt"

    phonemizer = Phonemizer.from_checkpoint(dp_model_filepath)
    phonemes = phonemizer(text, lang="ms")

    phonemes = collapse_whitespace(phonemes)

    return phonemes
