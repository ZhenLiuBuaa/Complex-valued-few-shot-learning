#-------------------------------------
# Project: Transductive Propagation Network for Few-shot Learning
# Date: 2019.1.11
# Author: Yanbin Liu
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
from    torch.nn import functional as F

from torch.autograd import Variable
import numpy as np
from complex_module import Complex,\
    C_MaxPooling, C_conv2d, C_BatchNorm2d,\
    C_ReLU, complex_weight_init, C_Linear, C_BatchNorm, C_AvePooling

class CNNEncoder(nn.Module):
    """Encoder for feature embedding"""
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args
        h_dim, z_dim = args['h_dim'], args['z_dim']
        if not self.args['complex']:
            if self.args['Relation_layer'] == 1:
                self.layer1 = nn.Sequential(
                                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
                self.layer2 = nn.Sequential(
                                nn.Conv2d(128,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
                self.layer3 = nn.Sequential(
                                nn.Conv2d(128,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
                self.layer4 = nn.Sequential(
                                nn.Conv2d(128,64,kernel_size=3,padding=1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
            else:
                # layer 1
                self.layer11 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),

                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer12 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer13 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer14 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                # layer 2
                self.layer21 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=5, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer22 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer23 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=5, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer24 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=5, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                # layer 1
                self.layer31 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer32 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer33 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
                self.layer34 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=7, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
        else:
            self.layer1 = nn.Sequential(
                C_conv2d(3, 64, kernel_size=3, padding=1),
                C_BatchNorm2d(64),
                C_ReLU(),
                #C_MaxPooling(2),
                C_AvePooling(2)
                )
            self.layer2 = nn.Sequential(
                C_conv2d(64, 64, kernel_size=3, padding=1),
                C_BatchNorm2d(64),
                C_ReLU(),
                #C_MaxPooling(2),
                C_AvePooling(2)
            )
            self.layer3 = nn.Sequential(
                C_conv2d(64, 64, kernel_size=3, padding=1),
                C_BatchNorm2d(64),
                C_ReLU(),
                #C_MaxPooling(2),
                C_AvePooling(2)
                )
            self.layer4 = nn.Sequential(
                C_conv2d(64, 64, kernel_size=3, padding=1),
                C_BatchNorm2d(64),
                C_ReLU(),
                #C_MaxPooling(2),
                C_AvePooling(2)
            )


    def forward(self,x):
        """x: bs*3*84*84 """
        if self.args['Relation_layer'] == 1:
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            return out
        else:
            out1 = self.layer11(x)
            out1 = self.layer12(out1)
            out1 = self.layer13(out1)
            out1 = self.layer14(out1)

            out2 = self.layer21(x)
            out2 = self.layer22(out2)
            out2 = self.layer23(out2)
            out2 = self.layer24(out2)

            out3 = self.layer31(x)
            out3 = self.layer32(out3)
            out3 = self.layer33(out3)
            out3 = self.layer34(out3)
            return [out1, out2, out3]




class RelationNetwork(nn.Module):
    """Graph Construction Module"""
    def __init__(self, args):
        super(RelationNetwork, self).__init__()
        self.args = args
        if self.args['complex']:
            self.layer1 = nn.Sequential(
                C_conv2d(64, 64, kernel_size=3, padding=1),
                C_BatchNorm2d(64),
                C_ReLU(),
                #C_MaxPooling(kernel_size=2, padding=1),
                C_AvePooling(kernel_size=2, padding=1)
                )
            self.layer2 = nn.Sequential(
                C_conv2d(64, 1, kernel_size=3, padding=1),
                C_BatchNorm2d(1),
                C_ReLU(),
                #C_MaxPooling(kernel_size=2, padding=1),
                C_AvePooling(kernel_size=2, padding=1),
                )

            self.fc3 = C_Linear(2 * 2, 8)
            self.fc4 = C_Linear(8, 1)
            self.relu = C_ReLU()
        else:

            self.layer1 = nn.Sequential(
                            nn.Conv2d(64,64,kernel_size=3,padding=1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, padding=1)

                            )
            self.layer2 = nn.Sequential(
                            nn.Conv2d(64,1,kernel_size=3,padding=1),
                            nn.BatchNorm2d(1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, padding=1))

            self.fc3 = nn.Linear(2*2, 8)
            self.fc4 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
            self.m0 = nn.MaxPool2d(2)            # max-pool without padding
            self.m1 = nn.MaxPool2d(2, padding=1) # max-pool with padding

    def forward(self, x, rn):

        x = x.view(-1, 64, 5, 5)
        
        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        #out = out.view(out.size(0),-1)

        out = self.relu(self.fc3(out))
        out = self.fc4(out) # no relu
        out = out.view(out.size(0), -1)
        # bs*1

        return out


class Prototypical(nn.Module):
    """Main Module for prototypical networlks"""
    def __init__(self, args):
        super(Prototypical, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)

    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        inp = torch.cat((support,query), 0)
        if self.args['complex']:
            inp = Complex()
        emb   = self.encoder(inp) # 80x64x5x5
        emb_s, emb_q = torch.split(emb, [num_classes*num_support, num_classes*num_queries], 0)
        emb_s = emb_s.view(num_classes, num_support, 1600).mean(1)
        emb_q = emb_q.view(-1, 1600)
        emb_s = torch.unsqueeze(emb_s,0)     # 1xNxD
        emb_q = torch.unsqueeze(emb_q,1)     # Nx1xD
        dist  = ((emb_q-emb_s)**2).mean(2)   # NxNxD -> NxN

        ce = nn.CrossEntropyLoss().cuda(0)
        loss = ce(-dist, torch.argmax(q_labels,1))
        ## acc
        pred = torch.argmax(-dist,1)
        gt   = torch.argmax(q_labels,1)
        correct = (pred==gt).sum()
        total   = num_queries*num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc


class LabelPropagation(nn.Module):
    """Label Propagation"""
    def __init__(self, args):
        super(LabelPropagation, self).__init__()
        self.im_width, self.im_height, self.channels = list(map(int, args['x_dim'].split(',')))
        self.h_dim, self.z_dim = args['h_dim'], args['z_dim']

        self.args = args
        self.encoder = CNNEncoder(args)
        self.relation = RelationNetwork(args)

        if   args['rn'] == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([args['alpha']], requires_grad=False).cuda(0)
        elif args['rn'] == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([args['alpha']]).cuda(0), requires_grad=True)
        if args['phrase']:
            if  args['beta'] ==30:
                self.beta = nn.Parameter(torch.tensor([args['phrase']]).cuda(0), requires_grad=True)
            else:
                self.beta = nn.Parameter(torch.tensor([args['phrase']]), requires_grad=False).cuda(0)
        #elif args['Beta'] ==300:
         #   self.BetaNet =
        self.bceloss = nn.BCELoss().cuda(0)
        self.CELoss = nn.CrossEntropyLoss().cuda(0)
    def forward(self, inputs):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        eps = np.finfo(float).eps

        [support, s_labels, query, q_labels] = inputs
        num_classes = s_labels.shape[1]
        num_support = int(s_labels.shape[0] / num_classes)
        num_queries = int(query.shape[0] / num_classes)

        # Step1: Embedding
        inp     = torch.cat((support,query), 0)
        if self.args['complex']:
            inp = Complex(inp)
        emb_all = self.encoder(inp)

        emb_all = emb_all.view(-1, 1600) if self.args['Relation_layer'] == 1 else [emb.view(-1, 1600) for emb in emb_all]
        (N, d) = (emb_all.shape[0], emb_all.shape[1]) if self.args['Relation_layer'] == 1 else (emb_all[0].shape[0], emb_all[0].shape[1])


        # Step2: Graph Construction
        ## sigmma
        if self.args['rn'] in [30, 300]:
            if not self.args['Relation_layer'] == 1:
                emb_all = torch.cat(emb_all, 0)
            self.sigma   = self.relation(emb_all , self.args)
            #self.sigma = 0.25
            ## W
            if self.args['complex']:

                emb_all.real = (emb_all.real*self.sigma.real + emb_all.imag*self.sigma.imag)/self.sigma.mag()
                emb_all.imag = (emb_all.imag*self.sigma.real - emb_all.real*self.sigma.imag)/self.sigma.mag()
            else:
                emb_all = emb_all / (self.sigma+eps) # N*d

            if self.args['center'] != 0:
                if self.args['complex']:
                    emb_support_real = emb_all.real[:len(support)].view(num_classes, num_support, -1)
                    emb_support_imag = emb_all.imag[:len(support)].view(num_classes, num_support, -1)
                    emb_query_real = emb_all.real[len(support):].view(num_classes, num_queries, -1)
                    emb_query_imag = emb_all.imag[len(support):].view(num_classes, num_queries, -1)
                    Center_emb_real = torch.cat([emb_support_real, emb_query_real], 1)
                    Center_emb_imag = torch.cat([emb_support_imag, emb_query_imag], 1)
                    even_emb_real = Center_emb_real.mean(1).unsqueeze(1)
                    even_emb_imag = Center_emb_imag.mean(1).unsqueeze(1)
                    Center_emb_real = Center_emb_real - even_emb_real
                    Center_emb_imag = Center_emb_imag - even_emb_imag
                    Center_emb = (Center_emb_real**2 + Center_emb_imag**2).mean(-1)
                    Center_emb = torch.exp(-Center_emb / 2)
                else:
                    emb_support = emb_all[:len(support)].view(num_classes, num_support, -1)
                    emb_query = emb_all[len(support):].view(num_classes, num_queries, -1)
                    Center_emb = torch.cat([emb_support, emb_query], 1)
                    even_emb = Center_emb.mean(1).unsqueeze(1)
                    Center_emb = ((Center_emb - even_emb)**2).mean(2)
                    Center_emb = torch.exp(-Center_emb/2)

                center_loss = self.bceloss(Center_emb, torch.ones(Center_emb.shape).cuda(0))

            '''if self.args['complex']:
                emb1 = emb_all.real.unsqueeze(0) - emb_all.real.unsqueeze(1)
                emb2 = emb_all.imag.unsqueeze(0) - emb_all.imag.unsqueeze(1)
                W = (emb1 ** 2 + emb2 ** 2).mean(-1)
            else:
                emb1 = torch.unsqueeze(emb_all.mag() if self.args['complex'] else emb_all, 1)  # N*1*d
                emb2 = torch.unsqueeze(emb_all.mag() if self.args['complex'] else emb_all, 0)  # 1*N*d
                W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N'''
            emb1 = torch.unsqueeze(emb_all.mag() if self.args['complex'] else emb_all, 1)  # N*1*d
            emb2 = torch.unsqueeze(emb_all.mag() if self.args['complex'] else emb_all, 0)  # 1*N*d
            W = ((emb1 - emb2) ** 2).mean(2)  # N*N*d -> N*N
            W = torch.softmax(W, -1)


            if self.args['phrase'] and self.args['complex']:
                #emb_all_real = emb_all.real.clone()
                #emb_all_imag = emb_all.imag.clone()
                sign = emb_all.real / (emb_all.imag +eps)
                sign = (sign>0).int()
                sign = sign + (emb_all.real<0).int()*2 + 1

                phrase_emb1 = sign.unsqueeze(0)
                phrase_emb2 = sign.unsqueeze(1)
                phrase_W = (phrase_emb1-phrase_emb2).abs()
                phrase_W[phrase_W==3]=1
                phrase_W = (phrase_W.float()**2).mean(-1)
                phrase_W = torch.softmax(phrase_W, -1)
                '''phrase_emb = emb_all.imag/(emb_all.mag()+eps)
                phrase_emb1 = phrase_emb.unsqueeze(0)
                phrase_emb2 = phrase_emb.unsqueeze(1)'''
                #phrase_W = ((phrase_emb1 - phrase_emb2)**2).mean(2)
            if self.args['phrase']:
                W = self.beta*W + (1-self.beta)*phrase_W

            if not self.args['Relation_layer'] == 1:
                W = W.view(N, self.args['Relation_layer'], N, self.args['Relation_layer'])
                W = W.transpose(1, 2)
                W = W.contiguous()
                W = W.view(N, N, -1)
                W = W.min(-1)[0]
            W       = torch.exp(-W/2)



        ## keep top-k values
        if self.args['k']>0:
            topk, indices = torch.topk(W, self.args['k'])
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = torch.eq((mask+torch.t(mask))>0).type(torch.float32)      # union, kNN graph
            #mask = ((mask>0)&(torch.t(mask)>0)).type(torch.float32)  # intersection, kNN graph
            W    = W*mask

        ## normalize
        D       = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1      = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2      = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S       = D1*W*D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        #ys = s_labels
        #yu = torch.zeros(num_classes*num_queries, num_classes).cuda(0)
        #yu = (torch.ones(num_classes*num_queries, num_classes)/num_classes).cuda(0)
        s_index = torch.argmax(s_labels, 1)
        q_index = torch.argmax(q_labels, 1).long().cuda(0)

        index = torch.cat([s_index, q_index], 0)
        y = self.label2edge(index)
            #get phrase-loss
        '''if self.args['phrase']:
            phrase_label = y.clone()
            phrase_loss = self.bceloss(phrase_W, phrase_label)'''
        y[num_classes * num_support:, :] = 0
        y[:, num_classes * num_support:] = 0
        y[num_classes * num_support:, num_classes * num_support:] = 1 / num_classes
        #############
        #y  = torch.cat((ys,yu),0)


        F  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), y)
        '''except:
            tmp = torch.eye(N).cuda(0)-self.alpha*S+eps
            tmp = torch.from_numpy(np.linalg.pinv(tmp.cpu().detach().numpy())).cuda(0)
            F = torch.matmul(tmp, y)'''
        F_q2s = F[num_classes * num_support:, :num_classes * num_support]
        F_q2s = F_q2s.view(F_q2s.shape[0], num_classes, num_support)
        F_q2s = F_q2s.sum(-1) / num_support
        Fq = F[num_classes*num_support:, :num_classes*num_support]  # query predictions
        Fq = Fq.view(-1, num_classes, num_support)
        Fq = Fq.sum(-1)
        Fq2q = F[num_classes * num_support:, num_classes * num_support:]
        Fq2q = Fq2q.view(Fq2q.shape[0], num_classes, num_queries)
        Fq2q = Fq2q.sum(-1) / num_queries
        # Step4: Cross-Entropy Loss


        ## both support and query loss
        q_gt = torch.argmax(q_labels, 1)
        q2q_gt = torch.argmax(q_labels, 1)

        loss = self.CELoss(F_q2s, q_gt) + self.CELoss(Fq2q, q2q_gt) + \
               ((self.args['center']*center_loss) if self.args['center'] else 0)
                #+ (self.args['phrase']*phrase_loss if self.args['phrase'] else 0)
        ## acc
        predq = torch.argmax(Fq,1)
        gtq   = torch.argmax(q_labels,1)
        correct = (predq==gtq).sum()
        total   = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc

    def label2edge(self, label):
        # get size
        num_samples = label.size(0)

        # reshape
        label_i = label.unsqueeze(-1).repeat(1, num_samples)
        label_j = label_i.transpose(0, 1)

        # compute edge
        edge = torch.eq(label_i, label_j).float().cuda(0)
        return edge

