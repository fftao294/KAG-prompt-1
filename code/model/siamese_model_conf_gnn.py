import torch.nn as nn
import torch
import torch.nn.functional as F

affine_par = True
from . import ConvGRU2 as ConvGRU


class CoattentionModel(nn.Module):
    def __init__(self, all_channel=1024, all_dim=16):
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim * all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel * 2, all_channel, kernel_size=3, padding=1, bias=True)#true
        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, all_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.conv_fusion = nn.Conv2d(all_channel * 3, all_channel, kernel_size=3, padding=1, bias=True)
        self.relu_fusion = nn.ReLU(inplace=True)
        self.prelu = nn.ReLU(inplace=True)
        self.relu_m = nn.ReLU(inplace=True)
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 5

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs1, inputs2, inputs3, inputs4):

        input_size = inputs1.size()[2:]
        batch_num = inputs1.size()[0]
        x1s = torch.zeros(batch_num, 1024, input_size[0], input_size[1]).cuda()
        x2s = torch.zeros(batch_num, 1024, input_size[0], input_size[1]).cuda()
        x3s = torch.zeros(batch_num, 1024, input_size[0], input_size[1]).cuda()
        x4s = torch.zeros(batch_num, 1024, input_size[0], input_size[1]).cuda()
        for ii in range(batch_num):
            input1 = inputs1[ii, :, :, :][None].contiguous().clone()
            input2 = inputs2[ii, :, :, :][None].contiguous().clone()
            input3 = inputs3[ii, :, :, :][None].contiguous().clone()
            input4 = inputs4[ii, :, :, :][None].contiguous().clone()
            for passing_round in range(self.propagate_layers):

                attention1 = self.conv_fusion(
                    torch.cat([self.generate_attention(input1, input2), self.generate_attention(input1, input3),
                               self.generate_attention(input1, input4)], 1))  # message passing with concat operation
                attention2 = self.conv_fusion(
                    torch.cat([self.generate_attention(input2, input1), self.generate_attention(input2, input3),
                               self.generate_attention(input2, input4)], 1))
                attention3 = self.conv_fusion(
                    torch.cat([self.generate_attention(input3, input1), self.generate_attention(input3, input2),
                               self.generate_attention(input3, input4)], 1))
                attention4 = self.conv_fusion(
                    torch.cat([self.generate_attention(input4, input1), self.generate_attention(input4, input2),
                               self.generate_attention(input4, input3)], 1))

                h_v1 = self.ConvGRU(attention1, input1)
                h_v2 = self.ConvGRU(attention2, input2)
                h_v3 = self.ConvGRU(attention3, input3)
                h_v4 = self.ConvGRU(attention4, input4)
                input1 = h_v1.clone()
                input2 = h_v2.clone()
                input3 = h_v3.clone()
                input4 = h_v4.clone()


                # print('attention size:', attention3[None].contiguous().size(), exemplar.size())
                if passing_round == self.propagate_layers - 1:
                    x1s[ii, :, :, :] = self.my_fcn(h_v1, inputs1[ii, :, :, :][None].contiguous())
                    x2s[ii, :, :, :] = self.my_fcn(h_v2, inputs2[ii, :, :, :][None].contiguous())
                    x3s[ii, :, :, :] = self.my_fcn(h_v3, inputs3[ii, :, :, :][None].contiguous())
                    x4s[ii, :, :, :] = self.my_fcn(h_v4, inputs4[ii, :, :, :][None].contiguous())



        return x1s, x2s, x3s, x4s

    def message_fun(self, input):
        input1 = self.conv_fusion(input)
        input1 = self.relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0] * fea_size[1])  # N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0] * fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t)
        A = torch.bmm(exemplar_corr, query_flat)

        B = F.softmax(torch.transpose(A, 1, 2), dim=1)
        exemplar_att = torch.bmm(query_flat, B).contiguous()


        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        input1_mask = self.gate_s(input1_mask)
        input1_att = input1_att * input1_mask

        return input1_att


    def my_fcn(self, input1_att, exemplar):
        input1_att = torch.cat([input1_att, exemplar], 1)
        input1_att = self.conv1(input1_att)
        input1_att = self.bn1(input1_att)
        input1_att = self.prelu(input1_att)
        return input1_att


def GNNNet():
    model = CoattentionModel()
    return model
