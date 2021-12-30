import model
from Langs import langs

import time


MAX_LENGTH = 10

encoder, decoder = model.create_model(MAX_LENGTH)

eng_word2index, fra_word2index, eng_index2word, fra_index2word = langs.read_langs()
snt = input()

start = time.time()
print(model.predict(encoder, decoder, snt, eng_word2index, fra_index2word, MAX_LENGTH))
print(time.time() - start)
