#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Machine Translation
# Assignment 5
# Author: Vera Bernhard

import sys

def change_seq(line):
    line = line.rstrip()
    line_list = line.split(' ')
    for token in reversed(line_list):
        yield token


def main():
    infile = sys.argv[1]
    with open("test.changed.de",'w', encoding='utf-8') as outfile:
        with open(infile, 'r', encoding='utf-8') as infile:
            for line in infile:
                tok_it = change_seq(line)
                for tok in tok_it:
                    outfile.write(tok)
                    outfile.write(' ')
                outfile.write('\n')


if __name__ == "__main__":
    main()