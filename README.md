# Malay-Gaming-Dataset

We attempt to generate corresponding text-audio pair for Malay Gaming Transcript, which is created via ChatGPT 3.5

There are 3 jupyter notebooks to provide detailed step-by-step phonemes analysis and instructions to generate Malay Gaming Transcripts:

| S/N | Notebook | Description |
| -- | -- | -- |
| 1. | gaming_terms.ipynb | &bull; Extract gaming terms from [Wikipedia](https://en.wikipedia.org/wiki/Glossary_of_video_game_terms) and other gaming websites. <br>&bull; Analyse the difference between phonemes generated via [eng_to_ipa](https://github.com/mphilli/English-to-IPA) and [espeak bootphon phonemizer](https://github.com/bootphon/phonemizer). |
| 2. | gaming_gpt.ipynb | &bull; Create mapping for gaming terms (mainly English words) to modified versions in order to produce proper pronunciation via [Malaya VITS TTS model](https://malaya-speech.readthedocs.io/en/latest/tts-vits.html). <br>&bull; Generate Malay gaming transcripts using the extracted gaming terms via ChatGPT 3.5 <br>&bull; Generate corresponding audio for  |
| 3. | gaming_gpt_phon.ipynb | &bull; Generate phonemes for the above Malay gaming transcripts. |

Note that:

1. Set up instructions and the required python modules are provided within the jupyter notebooks. Please use python version of at least 3.10.
2. We attempt to generate proper sound by modifying word (e.g. 'purchase' -> 'per chease') on a best effort basis. As such, the sound may not be ideal for some gaming terms.
