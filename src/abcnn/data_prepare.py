import re
import jieba
import random
import csv
from tensorflow.contrib import learn


class Data_Prepare(object):

    def readfile(self, filename):
        texta = []
        textb = []
        tag = []
        # with open(filename) as tsv_f:
        #     reader = csv.reader(tsv_f, delimiter='\t')
        #     for row in reader:
        #         texta.append(self.pre_processing(row[1]))
        #         textb.append(self.pre_processing(row[2]))
        #         tag.append(row[0])
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                texta.append(self.pre_processing(line[1]))
                textb.append(self.pre_processing(line[2]))
                tag.append(line[0])

        # shuffle
        index = [x for x in range(len(texta))]
        random.shuffle(index)
        texta_new = [texta[x] for x in index]
        textb_new = [textb[x] for x in index]
        tag_new = [tag[x] for x in index]

        type = list(set(tag_new))
        dicts = {}
        tags_vec = []
        for x in tag_new:
            if x not in dicts.keys():
                dicts[x] = 1
            else:
                dicts[x] += 1
            temp = [0] * len(type)
            temp[int(x)] = 1
            tags_vec.append(temp)
        print(dicts)
        return texta_new, textb_new, tags_vec

    def pre_processing(self, text):
        text = re.sub('（[^（.]*）', '', text)
        text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
        words = ' '.join(jieba.cut(text)).split(" ")
        words = [x for x in ''.join(words)]
        return ' '.join(words)

    def build_vocab(self, sentences, path):
        # lens = [len(sentence.split(" ")) for sentence in sentences]
        # max_length = max(lens)
        # print("max length ",max_length)
        vocab_processor = learn.preprocessing.VocabularyProcessor(12)
        vocab_processor.fit(sentences)
        vocab_processor.save(path)


if __name__ == '__main__':
    data_pre = data_prepare()
    data_pre.readfile('dataset/sent_pair/bq/train.tsv')