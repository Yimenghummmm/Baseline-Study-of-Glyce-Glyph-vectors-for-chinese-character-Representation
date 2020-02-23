# set the lstm hyperparameter
class TrainingConfig(object):
    batch_size = 64
    lr = 0.001
    epoches = 30
    print_step = 5


class LSTMConfig(object):
    emb_size = 128  
    hidden_size = 128  # lstm hidden size
