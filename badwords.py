import pandas as pd, numpy as np

with open('./bad_word_list.txt') as f:
    bad_words = f.read().split('\n')

bws = set(bad_words)
print(len(bws))
import enchant
d = enchant.Dict('en_US')
bws = set([bw for bw in list(bws) if d.check(bw)])
print(len(bws))