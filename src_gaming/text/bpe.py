"""
alternative BPE tokenizer model, which allows building custom tokenizer model
"""

import os

# fmt: off
def pre_tokenizer(input_file_path):
    """
    process the text file into pure text corpus, by removing the directories
    """
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    dir_path = os.path.dirname(input_file_path)
    output_file_path = os.path.join(dir_path, base_name + "_textonly.txt")

    with open(input_file_path, "r", encoding="utf-8") as input_file, \
        open(output_file_path, "w", encoding="utf-8") as output_file:
        for line in input_file:
            # Split the line at the pipe and keep the part after the pipe
            parts = line.strip().split("|")
            if len(parts) > 1:
                processed_text = parts[1].strip()  # Remove any leading/trailing whitespace
                output_file.write(processed_text + "\n")
    return output_file_path
# fmt: on


def bpe_pretrain(corpus_file_path, vocab_size):
    """
    train the bpe tokenizer model based on the text corpus
    save the tokenizer as a pickle file
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    # Initialize a tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Trainer for the tokenizer, limiting the vocabulary to the top K most frequent subwords
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size)

    # Train the tokenizer
    tokenizer.train(files=[corpus_file_path], trainer=trainer)

    # base_name = os.path.splitext(os.path.basename(corpus_file_path))[0]
    dir_path = os.path.dirname(corpus_file_path)
    save_path = os.path.join(dir_path + f"/bpe_{vocab_size}.json")

    # Save the tokenizer
    tokenizer.save(save_path)

    return tokenizer


def load_tokenizer(load_path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(load_path)


if __name__ == "__main__":
    input = r"D:\Dropbox\Private\Code\AIAP\tts-melayu\data\intermediate\test.txt"
    output = pre_tokenizer(input)

    _ = bpe_pretrain(output, 1000)
    tokenizer = load_tokenizer(r"D:\Dropbox\Private\Code\AIAP\tts-melayu\data\intermediate\bpe.json")

    test_sentence = "Saya pergi ke pasa, pada pagi hari."

    output = tokenizer.encode(test_sentence)
    decoded_text = tokenizer.decode(output.ids)

    print("output:", output)
    print("Token IDs:", output.ids)

    print("Decoded Text:", decoded_text)
