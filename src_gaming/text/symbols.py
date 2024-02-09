"""from https://github.com/keithito/tacotron

Defines the set of symbols used in text input to the model.
"""

PAD = "_"
PUNCTUATION = ';:,.!?¡¿—…"«»“” '
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LETTERS_IPA = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸ"
LETTERS_IPA = (
    LETTERS_IPA + "θœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# Use this and comment out the line for 'symbols' below if you intend to use
# Facebook research MMS-TTS pretrained zlm models
# fb_symbols = "yg feto5j_3k–ia0n6duc'hq-pmwr4slbz"
# symbols = list(fb_symbols)

# Export all symbols
# Integer id = 0 corresponding to white space i.e. symbols[0] = " "
symbols = [PAD] + list(PUNCTUATION) + list(LETTERS) + list(LETTERS_IPA)

# Special symbol ids
SPACE_ID = symbols.index(" ")
