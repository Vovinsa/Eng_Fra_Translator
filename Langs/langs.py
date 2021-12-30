import torch


def read_langs():
    eng_word2index = {}
    fra_word2index = {}

    eng_index2word = {0: "SOS", 1: "EOS"}
    fra_index2word = {}
    with open("Langs/eng_word2index.txt") as f:
        for word in f.readlines():
            key, val = word.strip().split(":")
            eng_word2index[key] = int(val)
            eng_index2word[int(val)] = key

    with open("Langs/fra_word2index.txt") as f:
        for word in f.readlines():
            key, val = word.strip().split(":")
            fra_word2index[key] = int(val)
            fra_index2word[int(val)] = key
    return eng_word2index, fra_word2index, eng_index2word, fra_index2word


def normalize_sentence(sentence: str):
    return sentence.lower()


def word2index(word: str, words_dict: dict):
    return words_dict[word]


def index2word(index: int, indexes_dict: dict):
    return indexes_dict[index]


def read_sentence(sentence: str, words_dict: dict):
    sentence = normalize_sentence(sentence)
    sentence = [word2index(word, words_dict) for word in sentence.split(" ")]
    sentence.append(1)
    return torch.tensor(sentence, dtype=torch.long)
