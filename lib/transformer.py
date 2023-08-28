import torch
import torch.nn as nn
import copy

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=2192, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)#(1936,8,0.1)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        #src=rel_input, input_key_padding_mask=masks
        #[最多的关系数量,图片数量,1936]，[图片数量,最多的关系数量] transfoemer的核心为self-attention
        #[L输入sequence的长度(例如一个句子的长度),N批大小(例如一个批的句子个数),E词向量长度]
        #把关系看成句子的每个字，这是合理的因为不同帧之间的关系有着联系，和NLP一样
        #把图片数量看成句子个数
        #1936作为特征长度
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)
        #ADD&Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        #FFN
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        #ADD&Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim=2192, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        #decoder层
        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)


        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):

        tgt2, global_attention_weights = self.multihead2(query=global_input+position_embed, key=global_input+position_embed,
                                                         value=global_input, key_padding_mask=input_key_padding_mask)
        #Add&Norm
        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        #FFN
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        #Add&Norm
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):#num_layers=1
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)#就等于一层encoder_layer
        self.num_layers = num_layers#1

    def forward(self, input, input_key_padding_mask):
        #input=rel_input, input_key_padding_mask=masks
        #[最多的关系数量,图片数量,1936]，[图片数量,最多的关系数量]
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, embed_dim):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)#num_layers=3
        self.num_layers = num_layers#3


    def forward(self, global_input, input_key_padding_mask, position_embed):

        output = global_input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):#只有最后一层的结果会被return
            output, global_attention_weights = layer(output, input_key_padding_mask, position_embed)
            weights[i] = global_attention_weights

        if self.num_layers>0:
            return output, weights
        else:
            return output, None


class transformer(nn.Module):
    ''' Spatial Temporal Transformer
        local_attention: spatial encoder
        global_attention: temporal decoder
        position_embedding: frame encoding (window_size*dim)
        mode: both--use the features from both frames in the window
              latter--use the features from the latter frame in the window
    '''
    def __init__(self, enc_layer_num=1, dec_layer_num=3, embed_dim=1936, nhead=8, dim_feedforward=2048,#此处修改embed_dim
                 dropout=0.1, mode=None):
        super(transformer, self).__init__()
        self.mode = mode

        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)#(1936,8,2048,0.1)  现在2448 8 2048 0.1
        self.local_attention = TransformerEncoder(encoder_layer, enc_layer_num)#enc_layer_num=1

        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                dropout=dropout)

        self.global_attention = TransformerDecoder(decoder_layer, dec_layer_num, embed_dim)

        self.position_embedding = nn.Embedding(2, embed_dim) #present and next frame
        nn.init.uniform_(self.position_embedding.weight)


    def forward(self, features, im_idx):
        #print(features.shape) #[rel_num, 1936] 现在[rel_num,2448]
        #print(im_idx)
        #tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,
         #       1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
         #       4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 6.,
         #       6., 6., 6., 6., 6., 6., 6., 6., 6., 7., 7., 7., 7., 7.,
         #       7., 7., 8., 8., 8., 8., 8., 8., 8., 8., 9., 9., 9., 9.,
         #       9., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 11.,
         #       11., 11., 11., 11., 11., 11., 12., 12., 12., 12., 12., 12., 13., 13.,
         #       13., 13., 13., 13., 14., 14., 14., 15., 15., 15., 15., 15., 15., 16.,
         #       16., 16., 16., 16., 16., 17., 17., 17., 18., 18., 18., 18., 18., 18.,
         #       18., 18., 18., 19., 19., 19., 19., 19., 19., 19., 19., 19., 19., 20.,
         #       20., 20., 20., 20., 20., 20., 20., 21., 21., 21., 21., 21., 21., 21.,
         #       22., 22., 22., 22., 23., 23., 23., 23., 23., 23., 23., 24., 24., 24.,
         #       24., 24., 24., 25., 25., 25., 25., 25., 26., 26., 26., 26., 26., 27.,
         #       27., 27., 27., 27., 27., 27., 28., 28., 28., 28., 29., 29., 29., 29.,
         #       29., 30., 30., 30., 30., 30., 31., 31., 31., 31., 31., 31., 31., 31.,
         #       32., 32., 32., 32., 32., 32., 32., 33., 33., 33., 34., 34., 34., 35.,
         #       35., 35., 35., 35., 36., 36., 36., 36., 37., 37., 37., 37., 37., 37.,
         #       37., 37.], device='cuda:0')

        rel_idx = torch.arange(im_idx.shape[0])#0到rel_num
        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame 12，就是一张图最多少关系
        b = int(im_idx[-1] + 1)#38，图片数量
        rel_input = torch.zeros([l, b, features.shape[1]]).to(features.device)
        #rel_input:[最多的关系数量,图片数量,1936] 现在[最多的关系数量,图片数量,2448]
        #mask:[38,12]
        masks = torch.zeros([b, l], dtype=torch.uint8).to(features.device)
        # TODO Padding/Mask maybe don't need for-loop
        for i in range(b):
            #i:从0到38
            #print(im_idx==i)
            #print(torch.sum(im_idx==i))
            #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #       device='cuda:0', dtype=torch.uint8)
            #tensor(9, device='cuda:0')
            rel_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]#把feature对应上
            masks[i, torch.sum(im_idx == i):] = 1
            #[0,9:],标记为1的位置说明没有这个框,就是说这个关系到此为止
        # spatial encoder
        #仅1层
        local_output, local_attention_weights = self.local_attention(rel_input, masks)
        local_output = (local_output.permute(1, 0, 2)).contiguous().view(-1, features.shape[1])[masks.view(-1) == 0]
        #print(local_output.shape)  [240, 1936]和输入的feature一样  [240,2448]

        global_input = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        #[24,37,1936] [关系*2,图片-1,1936]   现在[关系*2,图片-1,2448]
        position_embed = torch.zeros([l * 2, b - 1, features.shape[1]]).to(features.device)
        #[24,37,1936]  现在[24,37,2448]
        idx = -torch.ones([l * 2, b - 1]).to(features.device)
        #[24,37]
        idx_plus = -torch.ones([l * 2, b - 1], dtype=torch.long).to(features.device) #TODO
        #[24,37]

        # sliding window size = 2
        for j in range(b - 1):
            global_input[:torch.sum((im_idx == j) + (im_idx == j + 1)), j, :] = local_output[(im_idx == j) + (im_idx == j + 1)]
            #把j和j+1的特征给了global_input
            # 相当于现在把y轴的特征值扩大2倍
            #print((im_idx == j) + (im_idx == j + 1))
            #print(local_output[(im_idx == j) + (im_idx == j + 1)].shape)
            #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #       device='cuda:0', dtype=torch.uint8)
            #torch.Size([15, 1936])
            idx[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = im_idx[(im_idx == j) + (im_idx == j + 1)]#同理
            #print(torch.sum((im_idx == j) + (im_idx == j + 1)))
            #tensor(15, device='cuda:0')
            #tensor(11, device='cuda:0')
            #tensor(13, device='cuda:0')
            #tensor(17, device='cuda:0')
            #tensor(13, device='cuda:0')...
            #print(im_idx[(im_idx == j) + (im_idx == j + 1)])  [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.]
            #idx的每行代表global_input对应行的特征属于哪个图片
            idx_plus[:torch.sum((im_idx == j) + (im_idx == j + 1)), j] = rel_idx[(im_idx == j) + (im_idx == j + 1)]#同理  #TODO
            #print(rel_idx[(im_idx == j) + (im_idx == j + 1)]) tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

            #计算Ef
            position_embed[:torch.sum(im_idx == j), j, :] = self.position_embedding.weight[0]
            position_embed[torch.sum(im_idx == j):torch.sum(im_idx == j)+torch.sum(im_idx == j+1), j, :] = self.position_embedding.weight[1]

        #print(global_input.shape) torch.Size([24, 37, 1936])  现在[24, 37, 2448]
        global_masks = (torch.sum(global_input.view(-1, features.shape[1]),dim=1) == 0).view(l * 2, b - 1).permute(1, 0)
        #print(global_masks.shape) torch.Size([37, 24])
        # temporal decoder
        #3层
        global_output, global_attention_weights = self.global_attention(global_input, global_masks, position_embed)
        #print(global_output.shape)  torch.Size([24, 37, 1936])  现在[24, 37, 2448]
        #print(global_attention_weights.shape)  torch.Size([3, 37, 24, 24])

        output = torch.zeros_like(features)

        if self.mode == 'both':
            # both
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                if j == b - 2:
                    output[im_idx == j+1] = global_output[:, j][idx[:, j] == j+1]
                else:
                    output[im_idx == j + 1] = (global_output[:, j][idx[:, j] == j + 1] +
                                               global_output[:, j + 1][idx[:, j + 1] == j + 1]) / 2

        elif self.mode == 'latter':
            # later
            for j in range(b - 1):
                if j == 0:
                    output[im_idx == j] = global_output[:, j][idx[:, j] == j]

                output[im_idx == j + 1] = global_output[:, j][idx[:, j] == j + 1]

        #print(output.shape)  torch.Size([240, 1936]) 现在[240,2448]
        #print(global_attention_weights.shape)  torch.Size([3, 37, 24, 24])
        #print(local_attention_weights.shape)  torch.Size([1, 38, 12, 12])
        return output, global_attention_weights, local_attention_weights


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

