from itertools import zip_longest
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
from .config import TrainingConfig, LSTMConfig


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """Params：
            vocab_size: size of the vocabulary
            emb_size: size of the embedding
            hidden_size
            out_size: number of tags
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(2*hidden_size, out_size)

    def forward(self, sents_tensor, lengths):
        # [B, L, emb_size]
        emb = self.embedding(sents_tensor)  

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)

        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # [B, L, out_size]
        scores = self.lin(rnn_out)  

        return scores

    def test(self, sents_tensor, lengths, _):
        """ The last params is to match with the number of argument with bilstm_CRF"""
        
        # [B, L, out_size]
        logits = self.forward(sents_tensor, lengths)  
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """Params：
            vocab_size:
            emb_size:
            hidden_size：
            out_size:
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)
        
        #extra CRF Layer
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)

        # calculate CRF scores. [B, L, out_size, out_size]:
        # every character is matched with a [out_size, out_size] matrix
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)

        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """Viterbi algorithm"""

        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device

        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()

        # viterbi[i, j, k] the max score of the ith sentence and jth char and kth mark
        viterbi = torch.zeros(B, L, T).to(device)

        # backpointer[i, j, k]
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)

        # forward 
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # the previous tag of the first char is start_id 
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # backpropagate
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  #save the result
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

       
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        
        return tagids

class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size, crf=True):
        """ Train and Test the LSTM model
           Params:
            vocab_size: vocabulary size
            out_size: number of tags
            crf: choose to add crf layer (default set to True)
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda") 
            print("It is training on Cuda")
        else: 
            self.device = torch.device("cpu")

        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size
        self.crf = crf
        if not crf:
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss

        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,dev_word_lists, dev_tag_lists,word2id, tag2id):

        # sort the word and tag list by their length
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        B = self.batch_size

        print("##################################################################")
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_tags = tag_lists[ind:ind+B]

                losses += self.train_step(batch_sents,batch_tags, word2id, tag2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch: {}, step/total_step: {}/{} {:.2f}% \t| Loss:{:.4f}".format(
                        e, 
                        self.step, 
                        total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # test the model with the validation set
            val_loss = self.validate(dev_word_lists, dev_tag_lists, word2id, tag2id)

            print("Epoch: {}    \t\t\t\t|  Val Loss:{:.4f}".format(e, val_loss))
            print("##################################################################")


    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # prepare the dataset
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # calculate loss and update
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1

                # batchify
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # loss
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("Saving the best model.....")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """run the test set on the best model saved"""
        
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, lengths, tag2id)

     
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists

