#!/usr/bin/python

import argparse
import re
import sys
import time

from splitter import BigramSplitter

sys.setrecursionlimit(100000)


def split_text(train_files, source_file, destination_file):
    l_punctuation = '!"%…©»)]},.:;?'
    r_punctuation = "«([{#$*+<=>"
    no_punctuation = "-'’&×©/@\\^_`|~"

    # max_len = 29 as the longest word length according to the Belarusian language dictionary (2012)
    clf = BigramSplitter(max_len=29)
    t = time.time()
    clf.load_corpus('.', train_files)
    print('corpus loaded in ' time.time() - t 's')

    for phrase in source_file.readlines():
        words = clf.split(phrase)
        sent = ''
        last = '<S>'
        lot_dots = r'[.!?]\.\.'

        for w in words:
            if re.findall(lot_dots, w) or last in no_punctuation:
                sent += w
            elif w in l_punctuation or w in no_punctuation or last in r_punctuation:
                sent += w
            else:
                sent += ' ' + w
            last = w

        destination_file.write(sent.strip() + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits text into separate words')
    parser.add_argument('train', nargs='+', type=str,
                        help='Files that contains training samples')
    parser.add_argument('source', type=argparse.FileType('r'),
                        help='A file that contains words to be split')
    parser.add_argument('destination', type=argparse.FileType('w'),
                        help='A file for writing the result')

    args = parser.parse_args()
    split_text(args.train, args.source, args.destination)
