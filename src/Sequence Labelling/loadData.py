from os.path import join
from codecs import open

def build_corpus(split,  data_dir, make_vocab=True,):
    """load data"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        for line in f:
            word, tag = line.split("\t")
            word_list = word.split(' ')
            tag_list = tag.split(' ')
            word_lists.append(word_list)
            tag_lists.append(tag_list)
    
    # if make_vocab == True, vectorize the words and tags
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
