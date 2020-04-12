import torch
import subprocess as sub
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import os
import argparse
from models.encoder_vgg import Encoder
from models.decoder import Decoder
from models.attention import locationAttention as Attention
from models.seq2seq import Seq2Seq
from utils import visualizeAttn, writePredict, writeLoss, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP, load_data_func, tokens, loadData

parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

NUM_WORKER = 2
LAMBD = 1. # for grl, but don't use it. Use return_lambda instead!!
ALPHA = 1 # Don't use this one any more
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True
BATCH_SIZE = 32
learning_rate = 2 * 1e-4
lr_milestone = [20, 40, 60, 80, 100]

lr_gamma = 0.5

START_TEST = 1e4 # 1e4: never run test 0: run test from beginning
FREEZE = False
freeze_milestone = [65, 90]
EARLY_STOP_EPOCH = 30 # None: no early stopping
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512 # model/encoder.py SUM_UP=False: enc:dec = 1:2  SUM_UP=True: enc:dec = 1:1
CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
CurriculumModelID = args.start_epoch
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5
TEACHER_FORCING = False
MODEL_SAVE_EPOCH = 1

def return_lambda(epoch, start_epoch=15, end_epoch=60):
    return 1.
    #if epoch <= start_epoch:
    #    return 0.
    #elif epoch >= end_epoch:
    #    return 1.
    #else:
    #    return float(epoch - start_epoch) / (end_epoch - start_epoch)

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

log_softmax = torch.nn.LogSoftmax(dim=-1)
crit = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], 0.4)
# predict and gt follow the same shape of cross_entropy
# predict: 704, 83   gt: 704
def loss_label_smoothing(predict, gt):
    def smoothlabel_torch(x, amount=0.25, variance=5):
        mu = amount/x.shape[0]
        sigma = mu/variance
        noise = np.random.normal(mu, sigma, x.shape).astype('float32')
        smoothed = x*torch.from_numpy(1-noise.sum(1)).view(-1, 1).cuda() + torch.from_numpy(noise).cuda()
        return smoothed

    def one_hot(src): # src: torch.cuda.LongTensor
        ones = torch.eye(vocab_size).cuda()
        return ones.index_select(0, src)

    gt_local = one_hot(gt.data)
    gt_local = smoothlabel_torch(gt_local)
    loss_f = torch.nn.BCEWithLogitsLoss()
    gt_local = Variable(gt_local)
    res_loss = loss_f(predict, gt_local)
    return res_loss

def teacher_force_func(epoch):
    if epoch < 50:
        teacher_rate = 0.5
    elif epoch < 150:
        teacher_rate = (50 - (epoch-50)//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate

def teacher_force_func_2(epoch):
    if epoch < 200:
        teacher_rate = (100 - epoch//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate


def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER, pin_memory=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER, pin_memory=True)
    return train_loader, valid_loader, test_loader

def test_data_loader_batch(batch_size_nuevo):
    _, _, data_test = load_data_func()
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size_nuevo, shuffle=False, num_workers=NUM_WORKER, pin_memory=True)
    return test_loader

def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    train_domain = []
    for i in range(n_batch):
        if len(batch[i]) == 5: # valid and test
            idx, img, img_width, label, domain_l = batch[i]
            train_index.append(idx)
            train_in.append(img)
            train_in_len.append(img_width)
            train_out.append(label)
            train_domain.append(domain_l)
        else: # training
            idx, img, img_width, label, domain_l, idx2, img2, img_width2, label2, domain_l2 = batch[i]
            train_index.append(idx)
            train_index.append(idx2)
            train_in.append(img)
            train_in.append(img2)
            train_in_len.append(img_width)
            train_in_len.append(img_width2)
            train_out.append(label)
            train_out.append(label2)
            train_domain.append(domain_l)
            train_domain.append(domain_l2)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')
    train_domain = np.array(train_domain, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)
    train_domain = torch.from_numpy(train_domain)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    train_domain = train_domain[idx]
    return train_index, train_in, train_in_len, train_out, train_domain

def train(train_loader, seq2seq, opt, teacher_rate, epoch, lambd):
    seq2seq.train()
    total_loss = 0
    total_loss_d = 0
    for num, (train_index, train_in, train_in_len, train_out, train_domain) in enumerate(train_loader):
        train_in, train_out = Variable(train_in).cuda(), Variable(train_out).cuda()
        train_domain = Variable(train_domain).cuda()
        output, attn_weights, out_domain = seq2seq(train_in, train_out, train_in_len, lambd, teacher_rate=teacher_rate, train=True) # (100-1, 32, 62+1)
        batch_count_n = writePredict(epoch, train_index, output, 'train')
        train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
        output_l = output.view(-1, vocab_size) # remove last <EOS>

        if VISUALIZE_TRAIN:
            if 'e02-074-03-00,191' in train_index:
                b = train_index.tolist().index('e02-074-03-00,191')
                visualizeAttn(train_in.data[b,0], train_in_len[0], [j[b] for j in attn_weights], epoch, batch_count_n[b], 'train_e02-074-03-00')

        if LABEL_SMOOTH:
            loss = crit(log_softmax(output_l.view(-1, vocab_size)), train_label)
        else:
            loss = F.cross_entropy(output_l.view(-1, vocab_size),
                               train_label, ignore_index=tokens['PAD_TOKEN'])

        loss2 = F.cross_entropy(out_domain, train_domain)
        loss2 = ALPHA * loss2
        loss_total = loss + loss2
        opt.zero_grad()
        loss_total.backward()
        opt.step()
        total_loss += loss.data[0]
        total_loss_d += loss2.data[0]

    total_loss /= (num+1)
    total_loss_d /= (num+1)
    return total_loss, total_loss_d

def valid(valid_loader, seq2seq, epoch):
    seq2seq.eval()
    total_loss_t = 0
    total_loss_t2 = 0

    for num, (test_index, test_in, test_in_len, test_out, test_domain) in enumerate(valid_loader):
        lambd = LAMBD
        test_in, test_out = Variable(test_in, volatile=True).cuda(), Variable(test_out, volatile=True).cuda()
        test_domain = Variable(test_domain, volatile=True).cuda()
        output_t, attn_weights_t, out_domain_t = seq2seq(test_in, test_out, test_in_len, lambd, teacher_rate=False, train=False)
        batch_count_n = writePredict(epoch, test_index, output_t, 'valid')
        test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
        if LABEL_SMOOTH:
            loss_t = crit(log_softmax(output_t.view(-1, vocab_size)), test_label)
        else:
            loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                 test_label, ignore_index=tokens['PAD_TOKEN'])

        loss_t2 = F.cross_entropy(out_domain_t, test_domain)

        total_loss_t += loss_t.data[0]
        total_loss_t2 += loss_t2.data[0]

        if 'n04-015-00-01,171' in test_index:
            b = test_index.tolist().index('n04-015-00-01,171')
            visualizeAttn(test_in.data[b,0], test_in_len[0], [j[b] for j in attn_weights_t], epoch, batch_count_n[b], 'valid_n04-015-00-01')
    total_loss_t /= (num+1)
    total_loss_t2 /= (num+1)
    return total_loss_t, total_loss_t2

def test(test_loader, modelID, showAttn=True):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    model_file = 'save_weights/seq2seq-' + str(modelID) +'.model'
    pretrain_dict = torch.load(model_file)
    seq2seq_dict = seq2seq.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in seq2seq_dict}
    seq2seq_dict.update(pretrain_dict)
    seq2seq.load_state_dict(seq2seq_dict) #load
    print('Loading ' + model_file)

    seq2seq.eval()
    total_loss_t = 0
    start_t = time.time()
    for num, (test_index, test_in, test_in_len, test_out, test_domain) in enumerate(test_loader):
        lambd = LAMBD
        test_in, test_out = Variable(test_in, volatile=True).cuda(), Variable(test_out, volatile=True).cuda()
        test_domain = Variable(test_domain, volatile=True).cuda()
        output_t, attn_weights_t, out_domain_t = seq2seq(test_in, test_out, test_in_len, lambd, teacher_rate=False, train=False)
        batch_count_n = writePredict(modelID, test_index, output_t, 'test')
        test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
        if LABEL_SMOOTH:
            loss_t = crit(log_softmax(output_t.view(-1, vocab_size)), test_label)
        else:
            loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                test_label, ignore_index=tokens['PAD_TOKEN'])


        total_loss_t += loss_t.data[0]
        if showAttn:
            global_index_t = 0
            for t_idx, t_in in zip(test_index, test_in):
                visualizeAttn(t_in.data[0], test_in_len[0], [j[global_index_t] for j in attn_weights_t], modelID, batch_count_n[global_index_t], 'test_'+t_idx.split(',')[0])
                global_index_t += 1

    total_loss_t /= (num+1)
    writeLoss(total_loss_t, 'test')
    print('       TEST loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

def main(all_data_loader_func):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    if CurriculumModelID > 0:
        model_file = 'save_weights/seq2seq-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        pretrain_dict = torch.load(model_file)
        seq2seq_dict = seq2seq.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in seq2seq_dict}
        seq2seq_dict.update(pretrain_dict)
        seq2seq.load_state_dict(seq2seq_dict) #load
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)
    epochs = 5000
    if EARLY_STOP_EPOCH is not None:
        min_loss = 1e3
        min_loss_index = 0
        min_loss_count = 0

    if CurriculumModelID > 0:
        start_epoch = CurriculumModelID + 1
        for i in range(start_epoch):
            scheduler.step()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        # each epoch, random sample training set to be balanced with unlabeled test set
        train_loader, valid_loader, test_loader = all_data_loader_func()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        teacher_rate = teacher_force_func(epoch) if TEACHER_FORCING else False
        start = time.time()

        lambd = return_lambda(epoch)

        loss, loss_d = train(train_loader, seq2seq, opt, teacher_rate, epoch, lambd)
        writeLoss(loss, 'train')
        writeLoss(loss_d, 'domain_train')
        print('epoch %d/%d, loss=%.3f, domain_loss=%.3f, lr=%.6f, teacher_rate=%.3f, lambda_pau=%.3f, time=%.3f' % (epoch, epochs, loss, loss_d, lr, teacher_rate, lambd, time.time()-start))

        if epoch%MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(seq2seq.state_dict(), folder_weights+'/seq2seq-%d.model'%epoch)

        start_v = time.time()
        loss_v, loss_v_d = valid(valid_loader, seq2seq, epoch)
        writeLoss(loss_v, 'valid')
        writeLoss(loss_v_d, 'domain_valid')
        print('      Valid loss=%.3f, domain_loss=%.3f, time=%.3f' % (loss_v, loss_v_d, time.time()-start_v))

        test(test_loader, epoch, False) #~~~~~~

        if EARLY_STOP_EPOCH is not None:
            gt = loadData.GT_TE
            decoded = 'pred_logs/valid_predict_seq.'+str(epoch)+'.log'
            res_cer = sub.Popen(['./tasas_cer.sh', gt, decoded], stdout=sub.PIPE)
            res_cer = res_cer.stdout.read().decode('utf8')
            loss_v = float(res_cer)/100
            if loss_v < min_loss:
                min_loss = loss_v
                min_loss_index = epoch
                min_loss_count = 0
            else:
                min_loss_count += 1
            if min_loss_count >= EARLY_STOP_EPOCH:
                print('Early Stopping at: %d. Best epoch is: %d' % (epoch, min_loss_index))
                return min_loss_index

if __name__ == '__main__':
    print(time.ctime())
    mejorModelID = main(all_data_loader)
    print(time.ctime())
