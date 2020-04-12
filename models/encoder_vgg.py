import math
from torch import nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
#from models.vgg_tro_channel1 import vgg16_bn
from models.vgg_tro_channel3 import vgg16_bn, vgg19_bn

#torch.cuda.set_device(1)

DROP_OUT = False
LSTM = False
SUM_UP = True
PRE_TRAIN_VGG = True

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.lambd * grad_output.neg(), None


class Encoder(nn.Module):
    def __init__(self, hidden_size, height, width, bgru, step, flip):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.height = height
        self.width = width
        self.bi = bgru
        self.step = step
        self.flip = flip
        self.n_layers = 2
        self.dropout = 0.5
        self.domain_pool_size = [4, 2, 1]
        #self.layer = vgg16_bn(PRE_TRAIN_VGG)
        self.layer = vgg19_bn(PRE_TRAIN_VGG)

        if DROP_OUT:
            self.layer_dropout = nn.Dropout2d(p=0.5)
        if self.step is not None:
            #self.output_proj = nn.Linear((((((self.height-2)//2)-2)//2-2-2-2)//2)*128*self.step, self.hidden_size)
            self.output_proj = nn.Linear(self.height//16*512*self.step, self.height//16*512)

        if LSTM:
            RNN = nn.LSTM
        else:
            RNN = nn.GRU

        if self.bi: #8: 3 MaxPool->2**3    128: last hidden_size of layer4
            self.rnn = RNN(self.height//16*512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=True)
            if SUM_UP:
                self.enc_out_merge = lambda x: x[:,:,:x.shape[-1]//2] + x[:,:,x.shape[-1]//2:]
                self.enc_hidden_merge = lambda x: (x[0] + x[1]).unsqueeze(0)
        else:
            self.rnn = RNN(self.height//16*512, self.hidden_size, self.n_layers, dropout=self.dropout, bidirectional=False)

        self.domain_rnn = RNN(512, 512, 2, dropout=self.dropout, bidirectional=False)

        self.domain_classifier = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 100),
                nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, 2)
                )
        self.domain_classifier_sinBatchNorm = nn.Sequential(
                #nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                #nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 100),
                #nn.BatchNorm1d(100),
                nn.ReLU(),
                nn.Linear(100, 2)
                )

    # (32, 1, 80, 1400)
    def forward(self, in_data, in_data_len, lambd, hidden=None):
        batch_size = in_data.shape[0]
        out = self.layer(in_data) # torch.Size([32, 512, 4, 63])

        if DROP_OUT and self.training:
            out = self.layer_dropout(out)
        #out.register_hook(print)
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        out.contiguous()
        #out = out.view(-1, batch_size, (((((self.height-2)//2)-2)//2-2-2-2)//2)*128) # (t, b, f) (173, 32, 1024)
        out = out.view(-1, batch_size, self.height//16*512)
        if self.step is not None:
            time_step, batch_size, n_feature = out.shape[0], out.shape[1], out.shape[2]
            out_short = Variable(torch.zeros(time_step//self.step, batch_size, n_feature*self.step)).cuda() # t//STEP, b, f*STEP
            for i in range(0, time_step//self.step):
                part_out = [out[j] for j in range(i*self.step, (i+1)*self.step)]
                # reverse the image feature map
                out_short[i] = torch.cat(part_out, 1) # b, f*STEP

            out = self.output_proj(out_short) # t//STEP, b, hidden_size
        width = out.shape[0]
        src_len = in_data_len.numpy()*(width/self.width)
        src_len = src_len + 0.999 # in case of 0 length value from float to int
        src_len = src_len.astype('int')
        out = pack_padded_sequence(out, src_len.tolist(), batch_first=False)
        output, hidden = self.rnn(out, hidden)
        # output: t, b, f*2  hidden: 2, b, f
        output, output_len = pad_packed_sequence(output, batch_first=False)
        if self.bi and SUM_UP:
            output = self.enc_out_merge(output)
            #hidden = self.enc_hidden_merge(hidden)
       # # output: t, b, f    hidden:  b, f
        odd_idx = [1, 3, 5, 7, 9, 11]
        hidden_idx = odd_idx[:self.n_layers]
        final_hidden = hidden[hidden_idx]
        #if self.flip:
        #    hidden = output[-1]
        #    #hidden = hidden.permute(1, 0, 2) # b, 2, f
        #    #hidden = hidden.contiguous().view(batch_size, -1) # b, f*2
        #else:
        #    hidden = output[0] # b, f*2

        '''
            Domain flow
        '''
        reverse_out = GradReverse.apply(output, lambd) # reverse_out (15,32,512)
        #merged_reverse_out = reverse_out.mean(0) # 32,512
        reverse_out = pack_padded_sequence(reverse_out, output_len, batch_first=False)
        _, hidden_state = self.domain_rnn(reverse_out)

        merged_reverse_out = torch.cat((hidden_state[0], hidden_state[1]), 1) # 32, 1024

        if merged_reverse_out.shape[0] == 1: # batch size = 1
            domain_out = self.domain_classifier_sinBatchNorm(merged_reverse_out)
        else:
            domain_out = self.domain_classifier(merged_reverse_out)

        return output, final_hidden, domain_out # t, b, f*2    b, f*2


    # matrix: b, c, h, w    lens: list size of batch_size
    def conv_mask(self, matrix, lens):
        lens = np.array(lens)
        width = matrix.shape[-1]
        lens2 = lens * (width / self.width)
        lens2 = lens2 + 0.999 # in case le == 0
        lens2 = lens2.astype('int')
        matrix_new = matrix.permute(0, 3, 1, 2) # b, w, c, h
        matrix_out = Variable(torch.zeros(matrix_new.shape)).cuda()
        for i, le in enumerate(lens2):
            if self.flip:
                matrix_out[i, -le:] = matrix_new[i, -le:]
            else:
                matrix_out[i, :le] = matrix_new[i, :le]
        matrix_out = matrix_out.permute(0, 2, 3, 1) # b, c, h, w
        return matrix_out

class Spatial_Pyramid_Pool(nn.Module):
    def __init__(self, out_pool_size):
        super(Spatial_Pyramid_Pool, self).__init__()
        self.out_pool_size = out_pool_size

        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
    def forward(self, previous_conv):
        num_sample = previous_conv.shape[0]
        previous_conv_size = (previous_conv.shape[2], previous_conv.shape[3])
        # print(previous_conv.size())
        for i in range(len(self.out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / self.out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / self.out_pool_size[i]))
            h_pad = (h_wid*self.out_pool_size[i] - previous_conv_size[0] + 1)/2
            w_pad = (w_wid*self.out_pool_size[i] - previous_conv_size[1] + 1)/2
            h_pad = int(h_pad)
            w_pad = int(w_pad)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp

if __name__ == '__main__':
    print(vgg16_bn())
