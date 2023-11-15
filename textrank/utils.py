from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np



# 문장 한 줄을 기준으로 공백 split (Tokenize 커스텀)
def fn_split_space(sentence, data_type):

    # 불용어 정리
    list_stopword = ['질문', '사이', '관계', '출판']

    if data_type == 'content':
        list_stopword = ['옮긴이', '글쓴이', '제목', '목차', '미주', '해석', '책', '서론', '본론', '해설', '각주']
    
    list_word = sentence.split(',')
    list_return = list(map(lambda x: x.strip(), list_word))

    for stopword in list_stopword:
        list_return = [value for value in list_return if value != stopword]

    return list_return


def scan_vocabulary(sents, min_count=2, data_type='general'):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(str) returns list of str
    min_count : int
        Minumum term frequency

    Returns
    -------
    idx_to_vocab : list of str
        Vocabulary list
    vocab_to_idx : dict
        Vocabulary to index mapper.
    """
    counter = Counter(w for sent in sents for w in fn_split_space(sent, data_type))
    counter = {w:c for w,c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x:-x[1])]
    vocab_to_idx = {vocab:idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx

def tokenize_sents(sents, tokenize, data_type):
    """
    Arguments
    ---------
    sents : list of str
        Sentence list
    tokenize : callable
        tokenize(sent) returns list of str (word sequence)

    Returns
    -------
    tokenized sentence list : list of list of str
    """
    return [fn_split_space(sent, data_type) for sent in sents]

    # return [tokenize(sent) for sent in sents]

def vectorize(tokens, vocab_to_idx):
    """
    Arguments
    ---------
    tokens : list of list of str
        Tokenzed sentence list
    vocab_to_idx : dict
        Vocabulary to index mapper

    Returns
    -------
    sentence bow : scipy.sparse.csr_matrix
        shape = (n_sents, n_terms)
    """
    rows, cols, data = [], [], []
    for i, tokens_i in enumerate(tokens):
        for t, c in Counter(tokens_i).items():
            j = vocab_to_idx.get(t, -1)
            if j == -1:
                continue
            rows.append(i)
            cols.append(j)
            data.append(c)
    n_sents = len(tokens)
    n_terms = len(vocab_to_idx)
    x = csr_matrix((data, (rows, cols)), shape=(n_sents, n_terms))
    return x