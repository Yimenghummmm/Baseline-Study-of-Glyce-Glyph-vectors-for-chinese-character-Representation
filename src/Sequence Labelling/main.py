import time
from collections import Counter

from loadData import build_corpus
from util import extend_maps, prepocess_data_for_lstmcrf, Metrics, save_model
from model.crf import CRFModel
from model.bilstm_crf import BILSTM_Model


def crf_train_eval(train_data, test_data, remove_O=False):

    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    crf_model = CRFModel()
    crf_model.train(train_word_lists, train_tag_lists)
    save_model(crf_model, "./log/checkpoints/crf.pkl")

    print("##################################################################")
    print("Done Training, total time: {}s.".format(int(time.time()-start)))
    print("Evaluating CRF Model ...")

    pred_tag_lists = crf_model.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    model_name = "bilstm_crf" if crf else "bilstm"
    save_model(bilstm_model, "./log/checkpoints/"+model_name+".pkl")

    print("##################################################################")
    print("Done Training, total time: {}s.".format(int(time.time()-start)))
    print("Evaluating {} Model ...".format(model_name))

    pred_tag_lists, test_tag_lists = bilstm_model.test(test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()
    # print(classification_report(test_tag_lists, pred_tag_lists))

    return pred_tag_lists
###############################################################################
#                           Train                                             #
###############################################################################

#Load Data
print("Loading Data...")

train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir= "./data/ner/resume")
dev_word_lists, dev_tag_lists = build_corpus("dev", data_dir= "./data/ner/resume" , make_vocab=False)
test_word_lists, test_tag_lists = build_corpus("test", data_dir= "./data/ner/resume", make_vocab=False)
print("Done")

# CRF model 
print("Training with CRF Model...")
print("##################################################################")

crf_pred = crf_train_eval(
    (train_word_lists, train_tag_lists),
    (test_word_lists, test_tag_lists)
)


# CRF LSTM-CRF 
print("Training with BiLSTM-CRF Model...")
print("##################################################################")

# Needs to add <start> and <end> for decoding
crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)

# extra data preprocessing
train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

lstmcrf_pred = bilstm_train_and_eval(
    (train_word_lists, train_tag_lists),
    (dev_word_lists, dev_tag_lists),
    (test_word_lists, test_tag_lists),
    crf_word2id, crf_tag2id
)