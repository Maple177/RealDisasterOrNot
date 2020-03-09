import numpy as np
import pandas as pd
import nltk
import json
import time
import re
from collections import Counter
from nltk import sent_tokenize, TweetTokenizer
tweet_tokenizer = TweetTokenizer()


def clean_text(text):
    #clean text: eliminate @; numbers; hyperlinks
    pattern = re.compile('@\S*|http:\S*|https:\S*|http\S*|http://*S*')
    lines = [re.sub(pattern,'',line).strip() for line in text]
    lines = [re.sub(r'[^a-zA-Z?.!,¿\']+',' ',line).strip() for line in lines]

    #lines = [tweet_tokenizer.tokenize(line) for line in lines]
    return lines


class Text2mat():
    '''
    ATTETION: accept preprocessed strings(not tokenized)
    '''
    def __init__(self,lines=None,mode='word',max_len=None,pad='_PAD_',unk='_UNK_',
                set_unk_tokens=True,low_count_threshold=1,tokenizer=tweet_tokenizer):
        '''
        set_unk_tokens: turn some tokens into 'unseen' tokens to treat OOV problem.
        low_count_threshold: if set_unk_tokens is True, threshold to filter tokens with low frequency.
                             use unfrequent tokens as OOV tokens to train the embedding of 'UNK'. 
        '''
        self.max_len = max_len
        self.pad = pad
        self.unk = unk
        self.set_unk_tokens = set_unk_tokens
        self.low_count_thres = low_count_threshold
        self.tokenizer = tokenizer
        self.initialized = False
        
        #mode {word,char}
        assert lines is not None, "cannot initialize with empty lines"

        self.mode = mode

        if mode == 'word':
            lines = [tokenizer.tokenize(line) for line in lines]
        elif mode == 'char':
            lines = [list(line) for line in lines]
        else:
            raise NameError('mode unrecognized.')
        
        self._from_lines(lines)


    def _from_lines(self,lines):
        self.initialized = True

        tokens = list(set([self.pad,self.unk] + [token for line in lines for token in line]))

        if self.set_unk_tokens:
            self._add_unk(lines)
            tokens = [token for token in tokens if token not in self.low_frequented_tokens]
        
        self.tokens = tokens
        self.n_tokens = len(tokens)
        self.token_to_id = {token:i for i, token in enumerate(tokens)}
        self.id_to_token = {i:token for i, token in enumerate(tokens)}
        self.pad_ix = self.token_to_id[self.pad]
        self.unk_ix = self.token_to_id[self.unk]

    def _add_unk(self,lines):
        token_count = Counter()
        for line in lines:
            token_count.update(line)
        self.low_frequented_tokens = [token for token in token_count \
                                if token_count[token] < self.low_count_thres]

    def to_matrix(self,lines):
        assert self.initialized, "not initialized."
        assert self.mode in ['word','char'], "mode unrecognised."

        if self.mode == 'word':
            lines = [self.tokenizer.tokenize(line) for line in lines]
        elif self.mode == 'char':
            lines = [list(line) for line in lines]
        
        max_len = self.max_len or max(map(len,lines))
        matrix = np.full((len(lines),max_len),self.pad_ix)
        for i, line in enumerate(lines):
            if self.set_unk_tokens:
                row = []
                for token in line[:max_len]:
                    if token in self.low_frequented_tokens:
                        row.append(self.unk_ix)
                    else:
                        row.append(self.token_to_id.get(token,self.unk_ix))
            else:
                row = [self.token_to_id.get(token,self.unk_ix) for token in line[:max_len]]
            matrix[i,:len(row)] = row
        return matrix

    


