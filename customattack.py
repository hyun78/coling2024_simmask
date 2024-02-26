import pandas as pd, numpy as np
import enchant
from textbugger.bug import *
from textbugger.bugger import Bugger
from utils import parse_args
from dataset import get_preprocessed_dataframe

if __name__ == "__main__":

    with open('./bad_word_list.txt') as f:
        bad_words = f.read().split('\n')

    bws = set(bad_words)
    #print(len(bws))
    d = enchant.Dict('en_US')
    bws = set([bw for bw in list(bws) if d.check(bw)])
    #print(bws)


    bugs = [Insert(), Delete(), NeighborSwap(), SubstituteC()]

    bg = Bugger(bugs, bws)


    args = parse_args()

    df = get_preprocessed_dataframe(args, mode='test')

    df['bugged'] = df.apply(lambda x : bg(x['post']), axis=1)

    df['bugged_post'] = df.apply(lambda x : x['bugged'][0], axis=1)
    df['modified'] = df.apply(lambda x : x['bugged'][1], axis=1)
    df['orig_post'] = df['post']

    print(df[['post', 'bugged_post']])
    print(df[df['modified'] == True])


    modified = df[df['modified'] == True]

    bugged_and_target = modified[modified['targets'] == 1][['bugged_post', 'orig_post', 'targets']]

    bugged_and_target.to_csv(f"bugged_dataset/{args.test_dataset}_bugged.csv")
