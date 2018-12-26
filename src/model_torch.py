from __future__ import print_function
import sys
import torch
from torch import nn 
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
torch.manual_seed(1)    # reproducible
np.random.seed(1)


class Multihot_RNN(torch.nn.Module):
    def __init__(self, **config):
        #print('5=======')
        super(Multihot_RNN, self).__init__()
        #print('6=======')
        self.input_dim = config['input_dim']
        self.rnn_in_dim = config['rnn_in_dim']
        self.rnn_out_dim = config['rnn_out_dim']
        self.rnn_layer = config['rnn_layer']
        self.batch_first = config['batch_first']
        self.num_class = config['num_class']

        self.in1 = nn.Linear(self.input_dim, self.rnn_in_dim)
        self.rnn1 = nn.LSTM(
            input_size = self.rnn_in_dim, 
            hidden_size = int(self.rnn_out_dim / 2),
            num_layers = self.rnn_layer,
            batch_first = self.batch_first,
            bidirectional=True
            )      
        self.out1 = nn.Linear(self.rnn_out_dim, self.num_class)
        self.f1 = torch.sigmoid

    @property
    def rnn_out_dimen(self):
        return self.rnn_out_dim

    def forward_rnn(self, X_batch, X_len):
        batch_size = X_batch.shape[0]
        #X_batch = Variable(torch.from_numpy(X_batch).float())
        dd1 = sorted(range(len(X_len)), key=lambda k: X_len[k], reverse = True)
        dd = [0 for i in range(len(dd1))]
        for i,j in enumerate(dd1):
            dd[j] = i
        X_len_sort = list(np.array(X_len)[dd1])
        X_batch_v = X_batch[dd1]
        pack_X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_batch_v, X_len_sort, batch_first=True)

        ### Option I
        #X_out, _ = self.rnn1(pack_X_batch, None) 
        #unpack_X_out, _ = torch.nn.utils.rnn.pad_packed_sequence(X_out, batch_first=True)
        #indx = list(np.array(X_len_sort) - 1)
        #indx = [int(v) for v in indx]
        #X_out2 = unpack_X_out[range(batch_size), indx]

        ### Option II
        _,(X_out,_) = self.rnn1(pack_X_batch, None)
        X_out2 = torch.cat([X_out[0], X_out[1]], 1)

        X_out2 = X_out2[dd]    ## batch_size, HIDDEN_SIZE
        return X_out2


    '''
    def forward(self, X_in, X_len):
        batch_size, max_length, input_dim = X_in.shape 
        X_in = Variable(torch.from_numpy(X_in).float())
        assert len(X_len) == batch_size
        X_in = X_in.reshape(-1, input_dim)
        X_in = self.f1(self.in1(X_in))
        X_in = X_in.reshape(batch_size, max_length, -1)

        X_out = self.forward_rnn(X_in, X_len)
        return self.out1(X_out)  #### !!!!! F.softmax(xxx) don't need softmax 
        #return self.out1(X_out)  
    '''
    ### split forward into (i) forward_rnn_and_rnn_before (2) forward (after rnn)
    def forward_rnn_and_rnn_before(self, X_in, X_len):  ### this would be use afterward. 
        batch_size, max_length, input_dim = X_in.shape 
        X_in = Variable(torch.from_numpy(X_in).float())
        assert len(X_len) == batch_size
        X_in = X_in.reshape(-1, input_dim)
        X_in = self.f1(self.in1(X_in))
        X_in = X_in.reshape(batch_size, max_length, -1)
        X_out = self.forward_rnn(X_in, X_len)
        return X_out 

    def forward(self, X_in, X_len):
        X_out = self.forward_rnn_and_rnn_before(X_in, X_len)
        return self.out1(X_out)



class Multihot_Dictionary_RNN(Multihot_RNN, torch.nn.Module):
    ### new version
    def __init__(self, **config):
        #print('3=======')
        Multihot_RNN.__init__(self, **config)
        #print('4=======')
        self.fc1 = config['fc1']
        self.dictionary_size = config['dictionary_size']
        self.fc2 = config['fc2']
        self.lambda2 = config['lambda2']
        #######
        #self.Dictionary = Variable(torch.rand(self.rnn_out_dimen, self.dictionary_size), requires_grad = True)
        self.Dictionary = nn.Linear(self.rnn_out_dimen, self.dictionary_size)
        self.reconstruct_matrix = nn.Linear(self.rnn_out_dimen, self.input_dim)
        self.classify_matrix = nn.Linear(self.dictionary_size, self.num_class)
    

    ### new version 
    def dictionary_encoder(self, X_in):
        DTD = torch.matmul(self.Dictionary.weight, self.Dictionary.weight.transpose(1,0))   #### D is Dictionary;  D^T D
        #DTD = torch.matmul(self.Dictionary.transpose(1,0), self.Dictionary)   #### D is Dictionary;  D^T D
        DTD_inv = torch.inverse(DTD)        ### (D^T D)^{-1}
        DTD_inv_DT = torch.matmul(DTD_inv, self.Dictionary.weight)   ###  (D^T D)^{-1} D^T 
        #DTD_inv_DT = torch.matmul(DTD_inv, self.Dictionary.transpose(1,0))   ###  (D^T D)^{-1} D^T 
        assert DTD_inv_DT.requires_grad == True
        X_o = X_in.matmul(DTD_inv_DT.transpose(1,0))
        X_o = torch.clamp(X_o, min = 0)
        assert X_o.requires_grad == True 
        return X_o

    ### new version
    def dictionary_decoder(self, X):
        return torch.matmul(X, self.Dictionary.weight)
        #return torch.matmul(X, self.Dictionary.transpose(1,0))

    def reconstruct_after_decode(self, X):
        return torch.sigmoid(self.reconstruct_matrix(X))

    def forward(self, X_in, X_len):
        X_out = self.forward_rnn_and_rnn_before(X_in, X_len)
        code = self.dictionary_encoder(X_out)
        decode = self.dictionary_decoder(code)
        recon = self.reconstruct_after_decode(decode)
        classify_output = self.classify_matrix(code)
        return recon, classify_output, code 

    @property
    def check_dictionary(self):
        return self.Dictionary[1,3:7]

class faster_Multihot_Dictionary_RNN(Multihot_Dictionary_RNN):
    def __init__(self, **config):
        Multihot_Dictionary_RNN.__init__(self, **config)
        self.Dictionary = Variable(torch.rand(self.rnn_out_dimen, self.dictionary_size), requires_grad = True)

    def dictionary_encoder(self, X_in):
        DTD = torch.matmul(self.Dictionary.transpose(1,0), self.Dictionary)   #### D is Dictionary;  D^T D
        DTD_inv = torch.inverse(DTD)        ### (D^T D)^{-1}
        DTD_inv_DT = torch.matmul(DTD_inv, self.Dictionary.transpose(1,0))   ###  (D^T D)^{-1} D^T 
        assert DTD_inv_DT.requires_grad == True
        X_o = X_in.matmul(DTD_inv_DT.transpose(1,0))
        X_o = torch.clamp(X_o, min = 0)
        assert X_o.requires_grad == True 
        return X_o  

    def dictionary_decoder(self, X):
        return torch.matmul(X, self.Dictionary.transpose(1,0))


class Multihot_Pearl(Multihot_RNN, torch.nn.Module):
    def __init__(self, assignment, **config):
        Multihot_RNN.__init__(self, **config)
        self.batch_size = config['batch_size']
        self.assignment = assignment
        self.rule_size = len(self.assignment)
        self.out1 = nn.Linear(self.rule_size, self.num_class)
        self.prototype = Variable(torch.zeros(self.rule_size, self.rnn_out_dim), requires_grad = False)

    @staticmethod
    def normalize_by_column(T_2d):
        return T_2d / T_2d.norm(dim = 1, keepdim = True)

    def generate_single_prototype(self, X_in_assign, X_len_assign):
        leng = len(X_len_assign)
        num_of_iter = int(np.ceil(leng / self.batch_size))
        XOUT = Variable(torch.zeros(leng, self.rnn_out_dim))
        for i in range(num_of_iter):
            bgn, endn = i * self.batch_size, i * self.batch_size + self.batch_size
            X_in_batch = X_in_assign[bgn:endn]
            if X_in_batch.shape[0] == 0:    break 
            X_len_batch = X_len_assign[bgn:endn]
            X_out = self.forward_rnn_and_rnn_before(X_in_batch, X_len_batch)
            XOUT[bgn:endn] = X_out
            #XOUT = torch.cat([XOUT, X_out], 0) if i > 0 else X_out
        return XOUT.mean(0)

    def generate_prototype(self, X_in_all, X_len_all):
        prototype_vec = torch.zeros(self.rule_size, self.rnn_out_dim)
        for i,j in enumerate(self.assignment):
            X_in = X_in_all[j]
            X_len = [X_len_all[k] for k in j]
            prototype_vec[i,:] = self.generate_single_prototype(X_in, X_len)  
        self.prototype.data = prototype_vec
        self.prototype = self.normalize_by_column(self.prototype)


    def forward_prototype(self, X_in):
        ### normalize 
        X_in = self.normalize_by_column(X_in)
        return X_in.matmul(self.prototype.transpose(1,0))

    def forward(self, X_in, X_len):
        X_out = self.forward_rnn_and_rnn_before(X_in, X_len)
        Xp_out = self.forward_prototype(X_out)
        return self.out1(Xp_out)



class MNIST_base(nn.Module):
    def __init__(self, **config):
        #print('1=======')
        #print(super(MNIST_base, self))
        #super(MNIST_base, self).__init__()
        nn.Module.__init__(self)
        #print('2=======')
        self.input_dim = config['rows'] * config['cols']
        self.dim1 = config['dim1']
        self.dim2 = config['dim2']
        self.out_dim = config['num_class']

        self.fc0 = nn.Linear(self.input_dim, self.dim1)
        self.fc1 = nn.Linear(self.dim1, self.out_dim)


    def forward_fc(self, X):
        X = X.view(-1, self.input_dim)  ### 3D tensor => 2D tensor
        X1 = torch.sigmoid(self.fc0(X))
        return X1

    def forward(self, X, label):
        X1 = self.forward_fc(X)
        X2 = self.fc1(X1)
        return X2



class MNIST_dictionary(MNIST_base, Multihot_Dictionary_RNN):
    def __init__(self, **config):
        MNIST_base.__init__(self, **config)
        self.dictionary_size = config['dictionary_size']
        self.Dictionary = nn.Linear(self.dim1, self.dictionary_size)
        self.reconstruct_matrix = nn.Linear(self.dim1, self.input_dim)
        self.classify_matrix = nn.Linear(self.dictionary_size, self.out_dim)

    def forward(self, X, label):
        X1 = self.forward_fc(X)
        code = self.dictionary_encoder(X1)
        decode = self.dictionary_decoder(code)
        recon = self.reconstruct_after_decode(decode) ##### sigmoid !!!
        recon = torch.sigmoid(recon)
        classify_output = self.classify_matrix(code)    
        return recon, classify_output, code 


if __name__ == '__main__':
    from config import get_multihot_rnn_config, get_multihot_dictionary_rnn_config
    #from stream import max_length, batch_size, TrainFile, TestFile, Create_Multihot_Data
    #from stream import admis_dim as input_dim
    #from stream import batch_size as global_batch_size
    configure = get_multihot_dictionary_rnn_config()
    #RNN = Multihot_RNN(**configure)
    RNN = Multihot_Dictionary_RNN(**configure)
    print(RNN.rnn_out_dimen)








