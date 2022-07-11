import os
import sys
import glob
import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pytorch3d.ops import knn_points, knn_gather

from .utils import transform_to_global_KITTI

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers

def pairwise_distance_batch(x,y):
    """ 
        pairwise_distance
        Args:
            x: Input features of source point clouds. Size [B, c, N]
            y: Input features of source point clouds. Size [B, c, M]
        Returns:
            pair_distances: Euclidean distance. Size [B, N, M]
    """
    xx = torch.sum(torch.mul(x,x), 1, keepdim = True)#[b,1,n]
    yy = torch.sum(torch.mul(y,y),1, keepdim = True) #[b,1,n]
    inner = -2*torch.matmul(x.transpose(2,1),y) #[b,n,n]
    pair_distance = xx.transpose(2,1) + inner + yy #[b,n,n]
    device = torch.device('cuda')
    zeros_matrix = torch.zeros_like(pair_distance,device = device)
    pair_distance_square = torch.where(pair_distance > 0.0,pair_distance,zeros_matrix)
    error_mask = torch.le(pair_distance_square,0.0)
    pair_distances = torch.sqrt(pair_distance_square + error_mask.float()*1e-16)
    pair_distances = torch.mul(pair_distances,(1.0-error_mask.float()))
    return pair_distances

class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, x):
        return self.mlp.forward(x)

# Feature extraction: PointNet ----> maps individual points to a higher dimensional space
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=32):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    
    knn_points = x.view(batch_size * num_points, -1)[idx, :]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)#[b, n, k, 3],knn
    
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature, idx, knn_points


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x, idx, knn_pts = get_graph_feature(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x, idx, knn_pts


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()

        #src_embedding_self = self.model(src, src, None, None).transpose(2, 1).contiguous()
        #tgt_embedding_self = self.model(tgt, tgt, None, None).transpose(2, 1).contiguous()

        src_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        tgt_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()

        #src_embedding = src_embedding_self + src_embedding_cross
        #tgt_embedding = tgt_embedding_self + tgt_embedding_cross

        return src_embedding, tgt_embedding

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(3, args.dim, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(args.dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(args.dim, args.dim * 2, kernel_size=(3,1), bias=True, padding=(1,0)),
            nn.BatchNorm2d(args.dim * 2),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model2 = nn.Sequential(
            nn.Conv2d(args.dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.model3 = nn.Sequential(
            nn.Conv2d(args.dim * 2, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True))

        self.model4 = nn.Sequential(
            nn.Conv1d(16, 8, kernel_size=1, bias=True),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(8, 1, kernel_size=1, bias=True),
            #nn.Tanh(),
        )

        self.tah = nn.Tanh()

    def forward(self, x, y):
        """ 
            Inlier Evaluation.
            Args:
                x: Source neighborhoods. Size [B, N, K, 3]
                y: Pesudo target neighborhoods. Size [B, N, K, 3]
            Returns:
                x: Inlier confidence. Size [B, 1, N]
        """
        b, n, k, _ = x.size()

        x_1x3 = self.model1(x.permute(0,3,2,1)).permute(0,1,3,2)

        y_1x3 = self.model1(y.permute(0,3,2,1)).permute(0,1,3,2) # [b, n, k, 3]-[b, c, k, n]-->[b, c, n, k]
        
        x2 = x_1x3 - y_1x3 # Eq. (5)
        
        x = self.model2(x2) # [b, c, n, k]
        weight = self.model3(x2) # [b, c, n, k] 
        weight = torch.softmax(weight, dim=-1) # Eq. (6)
        x = (x * weight).sum(-1) # [b, c, n]
        x = 1 - self.tah(torch.abs(self.model4(x)))
        return x


def get_knn_index(x, k):
    """ 
        knn-graph.
        Args:
            x: Input point clouds. Size [B, 3, N]
            k: Number of nearest neighbors.
        Returns:
            idx: Nearest neighbor indices. Size [B * N * k]
            idx2: Nearest neighbor indices. Size [B, N, k]
    """
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    idx2 = idx
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    return idx, idx2


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.num_keypoints = args.num_keypoints
        self.k = args.k
        self.weight_function = Discriminator(args)

    def forward(self, *input):
        
        src = input[0]            # P_s ---> bx3xN
        tgt = input[1]            # P_t ---> bx3xN
        scores = input[2]
        src_idx = input[3]
        src_knn = input[4]

        k = 32

        batch_size = src.size(0)  # b
        num_dims_src = src.size(1) # 3
        N = src.size(2)           # N

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())        # P_t

        ############################## Inlier Evaluation Module ##############################
        # neighborhoods of pseudo target point clouds
        src_knn_corr = src_corr.transpose(2,1).contiguous().view(batch_size * N, -1)[src_idx, :]
        src_knn_corr = src_knn_corr.view(batch_size, N, k, num_dims_src)#[b, n, k, 3]

        # edge features of the pseudo target neighborhoods and the source neighborhoods 
        knn_distance = src_corr.transpose(2,1).contiguous().unsqueeze(2) - src_knn_corr #[b, n, k, 3]
        src_knn_distance = src.transpose(2,1).contiguous().unsqueeze(2) - src_knn #[b, n, k, 3]
        
        # inlier confidence
        conf = self.weight_function(knn_distance, src_knn_distance)#[b, 1, n] # Eq. (7)

        src2 = (src * conf).sum(dim = 2, keepdim = True) / conf.sum(dim = 2, keepdim = True)
        src_corr2 = (tgt * conf).sum(dim = 2, keepdim = True)/conf.sum(dim = 2,keepdim = True)
        src_centered = src - src2
        src_corr_centered = tgt - src_corr2
        H = torch.matmul(src_centered * conf, src_corr_centered.transpose(2, 1).contiguous())

        ########################### Get top k keypoints #########################
        
        conf, src_topk_idx = torch.topk(conf, k = self.num_keypoints, dim = 2, sorted=False)
        src_keypoints_idx_2 = src_topk_idx.repeat(1, N, 1) # [b, m, num_keypoints]
        src_keypoints_idx_3 = src_topk_idx.unsqueeze(-1).repeat(1, 3, 1, self.k) #[b, 3, num_keypoints, k]
        tgt_keypoints_knn = torch.gather(knn_distance.permute(0,3,1,2), dim = 2, index = src_keypoints_idx_3) #[b, 3, num_kepoints, k]

        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)

        # Top key points
        src_keypoints = torch.gather(src, dim = 2, index = src_keypoints_idx)
        tgt_keypoints = torch.gather(src_corr, dim = 2, index = src_keypoints_idx)

        
        
        #---------------Compute Rigid Body Transformation------------------

        R = []
        for i in range(src.size(0)):
    
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src2.mean(dim=2, keepdim=True)) + src_corr2.mean(dim=2, keepdim=True)

        src_keypoints_idx = src_topk_idx.repeat(1, N, 1) # [b, m, num_keypoints]
        
        src_transformed = transform_to_global_KITTI(R, t.view(batch_size, 3), src.transpose(2, 1).contiguous())
        src_transformed_knn_corr = src_transformed.transpose(2,1).contiguous().view(batch_size * N, -1)[src_idx, :]
        src_transformed_knn_corr = src_transformed_knn_corr.view(batch_size, N, k, num_dims_src) #[b, n, k, 3]
        knn_distance2 = src_transformed.unsqueeze(2) - src_transformed_knn_corr #[b, n, k, 3]
        src_keypoints_knn = torch.gather(knn_distance2.permute(0,3,1,2), dim = 2, index = src_keypoints_idx_3) #[b, 3, num_kepoints, k]

        return R, t.view(batch_size, 3), src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn

class LNet2(nn.Module):
    def __init__(self, args) -> None:
        super(LNet2, self).__init__()
        self.emb_dims = args.emb_dims

        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            print('DGCNN feature extraction')
            self.emb_nn = DGCNN(emb_dims=self.emb_dims)
        self.pointer = Transformer(args=args)
        
        self.head = SVDHead(args=args)


    def forward(self, pc_pair):
        src = pc_pair[:, 0, :, :]    #<bx1xNx3>
        src = src.squeeze(1)         #<bxNx3>
        tgt = pc_pair[:, 1, :, :]    #<bx1xNx3>
        tgt = tgt.squeeze(1)         #<bxNx3>

        rotation_ab = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(src.shape[0], 1, 1)
        translation_ab = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(src.shape[0], 1)
        
        src_embedding, src_idx, src_knn_pts = self.emb_nn(src.transpose(2,1).contiguous())   #<bxdxN>
        tgt_embedding, _, _ = self.emb_nn(tgt.transpose(2,1).contiguous())   #<bxdxN>

        src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        distance_map = pairwise_distance_batch(src_embedding, tgt_embedding) #[b, n, m]

        # point-wise matching map
        scores = torch.softmax(-distance_map, dim=2) #[b, n, n]  Eq. (1)

        rotation_ab, translation_ab, src_keypoints, tgt_keypoints, src_keypoints_knn, tgt_keypoints_knn = self.head(src.transpose(2,1).contiguous(), tgt.transpose(2,1).contiguous(), scores, src_idx, src_knn_pts)
        
        return rotation_ab, translation_ab, src_keypoints.transpose(2,1).contiguous(), tgt_keypoints.transpose(2,1).contiguous(), src_keypoints_knn, tgt_keypoints_knn
