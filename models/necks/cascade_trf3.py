import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # self.class_token_pos = nn.Parameter(torch.zeros(1, 1, num_pos_feats * 2))
        # self.class_token_pos

    def forward(self, x, class_token=False):
        # x = tensor_list.tensors
        # x: b,d,h,w
        num_feats = x.shape[1]
        num_pos_feats = num_feats // 2
        # mask = tensor_list.mask
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device).to(torch.bool)
        batch = mask.shape[0]
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-5
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        '''
        pos_x: b ,h, w, d//2
        pos_y: b, h, w, d//2
        pos: b, h*w, d
        '''
        if class_token:
            # pos = torch.cat((self.class_token_pos.repeat(batch, 1, 1), pos), dim=1)
            # pos = torch.cat((torch.mean(pos, dim=1, keepdim=True), pos), dim=1)
            pos = torch.cat((torch.zeros(batch, 1, pos.shape[2], dtype=pos.dtype, device=pos.device), pos), dim=1)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer2(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoderLayer3(nn.Module):
    ## use BN replace LN ##
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.BatchNorm2d(d_model)
        self.norm3 = nn.BatchNorm2d(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask = None,
                     memory_mask = None,
                     tgt_key_padding_mask = None,
                     memory_key_padding_mask = None,
                     pos = None,
                     query_pos = None,
                     shape = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        size = tgt.shape
        # tgt = self.norm2(tgt)
        tgt = self.norm2(tgt.permute(1,2,0).reshape(size[1], size[2], shape[0], shape[1])).flatten(2).permute(2,0,1)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        tgt = self.norm3(tgt.permute(1, 2, 0).reshape(size[1], size[2], shape[0], shape[1])).flatten(2).permute(2, 0, 1)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask = None,
                    memory_mask = None,
                    tgt_key_padding_mask = None,
                    memory_key_padding_mask = None,
                    pos = None,
                    query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransFusion(nn.Module):
    def __init__(self, in_chans1, in_chans2, d_model, nhead, dim_feedforward, patch_size=(12, 12),
                 dropout=0.1, activation='relu', normalize_before=False):
        super(TransFusion, self).__init__()
        self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.pose_encoding = PositionEmbeddingSine(normalize=True)
        self.patch_size = patch_size
        self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Conv2d(in_chans2, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x1, x2):
        # x1: input
        # x2: output
        x1 = self.to_patch1(x1)
        x2 = self.to_patch2(x2)
        b, d, h, w = x2.shape
        pos1 = self.pose_encoding(x1).transpose(0,1)
        pos2 = self.pose_encoding(x2).transpose(0,1)
        src = x1.flatten(2).permute(2,0,1)
        memory = self.tr_encoder(src=src, pos=pos1)
        tgt = x2.flatten(2).permute(2,0,1)
        out = self.tr_decoder(tgt=tgt, memory=memory, pos=pos1, query_pos=pos2)
        return out.permute(1,2,0).reshape(b, d, h, w)

class TransFusion2(nn.Module):
    def __init__(self, in_chans1_list, in_chans2, d_model, nhead, dim_feedforward, patch_size=(12, 12),
                 dropout=0.1, activation='relu', normalize_before=False):
        super(TransFusion2, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.pose_encoding = PositionEmbeddingSine(normalize=True)
        self.patch_size = patch_size
        self.to_patch1 = nn.ModuleList()
        for in_chans1 in in_chans1_list:
            self.to_patch1.append(nn.Sequential(
                                    nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size),
                                    nn.BatchNorm2d(d_model),
                                                )
                                  )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size),
                                    nn.BatchNorm2d(d_model),
                                                )
        self.out_conv = nn.Conv2d(in_chans2 + d_model, in_chans2, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        patchs = []
        pos1 = []
        for i in range(len(x1)):
            patchs.append(self.to_patch1[i](x1[i]))
        # x1 = self.to_patch1(x1)
        tmp = self.to_patch2(x2)
        b, d, h, w = tmp.shape
        for patch in patchs:
            pos1.append(self.pose_encoding(patch).transpose(0,1))
        # pos1 = self.pose_encoding(x1).transpose(0,1)
        patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(tmp).transpose(0,1)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        tgt = tmp.flatten(2).permute(2,0,1)
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=patchs, pos=pos1, query_pos=pos2)
        out = out.permute(1,2,0).reshape(b, d, h, w)
        out = self.out_conv(torch.cat(
            (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        return out

class TransFusion3(nn.Module):
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False):
        super(TransFusion3, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.pose_encoding = PositionEmbeddingSine(normalize=True)
        # self.patch_size = patch_size
        self.to_patch1 = nn.Sequential(
                                    nn.Conv2d(in_chans1, d_model, kernel_size=patch_size1, stride=patch_size1),
                                    nn.BatchNorm2d(d_model),
                                                )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(
                                    nn.Conv2d(in_chans2, d_model, kernel_size=patch_size2, stride=patch_size2),
                                    nn.BatchNorm2d(d_model),
                                                )
        self.out_conv = nn.Sequential(
                                    nn.Conv2d(d_model, in_chans2, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(in_chans2),
                                    nn.ReLU()
                                                )
        self.out_conv2 = nn.Sequential(
            nn.Conv2d(in_chans1, in_chans2, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_chans2),
            nn.ReLU()
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)


    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        # patchs = []
        # pos1 = []
        # for i in range(len(x1)):
        #     patchs.append(self.to_patch1[i](x1[i]))
        k_patch = self.to_patch1(x1)
        q_patch = self.to_patch2(x2)
        b, d, h, w = q_patch.shape
        # for patch in patchs:
        #     pos1.append(self.pose_encoding(patch).transpose(0,1))
        pos1 = self.pose_encoding(k_patch).transpose(0,1)
        # pos1 = torch.zeros(h*w, b, d, dtype=tmp.dtype, device=tmp.device)
        # patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        # pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(q_patch).transpose(0,1)
        # pos2 = torch.zeros(h*w, b, d, dtype=patchs.dtype, device=patchs.device)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        k_patch = k_patch.flatten(2).permute(2,0,1)
        tgt = q_patch.flatten(2).permute(2,0,1)
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=k_patch, pos=pos1, query_pos=pos2)
        # out = self.tr_decoder(tgt=patchs, memory=tgt, pos=pos2, query_pos=pos1)
        out = out.permute(1,2,0).reshape(b, d, h, w)
        # out = self.out_conv(torch.cat(
        #     (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        # out = F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + x2
        out = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + \
              F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        return out

class TransFusion4(nn.Module): # concat
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin'):
        super(TransFusion4, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn() ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero() ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        if overlap:
            stride_size1 = (patch_size1[0] // 2, patch_size1[1] // 2)
            stride_size2 = (patch_size2[0] // 2, patch_size2[1] // 2)
        else:
            stride_size1 = patch_size1
            stride_size2 = patch_size2
        self.to_patch1 = nn.Sequential(
                                    nn.AvgPool2d(kernel_size=patch_size1, stride=stride_size1),
                                    nn.Conv2d(in_chans1, d_model, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(d_model),
                                                )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(
                                    nn.AvgPool2d(kernel_size=patch_size2, stride=stride_size2),
                                    nn.Conv2d(in_chans2, d_model, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(d_model),
                                                )
        self.out_conv = ConvModule(
            d_model + in_chans1,
            in_chans2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        # patchs = []
        # pos1 = []
        # for i in range(len(x1)):
        #     patchs.append(self.to_patch1[i](x1[i]))
        k_patch = self.to_patch1(x1)
        q_patch = self.to_patch2(x2)
        b, d, h, w = q_patch.shape
        # for patch in patchs:
        #     pos1.append(self.pose_encoding(patch).transpose(0,1))
        pos1 = self.pose_encoding(k_patch).transpose(0,1)
        # pos1 = torch.zeros(h*w, b, d, dtype=tmp.dtype, device=tmp.device)
        # patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        # pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(q_patch).transpose(0,1)
        # pos2 = torch.zeros(h*w, b, d, dtype=patchs.dtype, device=patchs.device)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        k_patch = k_patch.flatten(2).permute(2,0,1)
        tgt = q_patch.flatten(2).permute(2,0,1)
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=k_patch, pos=pos1, query_pos=pos2)
        # out = self.tr_decoder(tgt=patchs, memory=tgt, pos=pos2, query_pos=pos1)
        out = out.permute(1,2,0).reshape(b, d, h, w)
        # out = self.out_conv(torch.cat(
        #     (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        # out = F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + x2
        out = self.out_conv(
            torch.cat((F.interpolate(x1, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True),
                       F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1)
            )
        return out

class TransFusion5(nn.Module): # add
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin', adapt_size1=(10,10), adapt_size2=(10,10)):
        super(TransFusion5, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingLearn() ### surported soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero() ### surported soon
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        if overlap:
            stride_size1 = (patch_size1[0] // 2, patch_size1[1] // 2)
            stride_size2 = (patch_size2[0] // 2, patch_size2[1] // 2)
        else:
            stride_size1 = patch_size1
            stride_size2 = patch_size2
        self.to_patch1 = nn.Sequential(
                                    #nn.AvgPool2d(kernel_size=patch_size1, stride=stride_size1),
                                    nn.AdaptiveAvgPool2d(adapt_size1),
                                    nn.Conv2d(in_chans1, d_model, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(d_model),
                                                )
        # self.to_patch1 = nn.Conv2d(in_chans1, d_model, kernel_size=patch_size, stride=patch_size)
        self.to_patch2 = nn.Sequential(
                                    #nn.AvgPool2d(kernel_size=patch_size2, stride=stride_size2),
                                    nn.AdaptiveAvgPool2d(adapt_size2),
                                    nn.Conv2d(in_chans2, d_model, kernel_size=1, stride=1),
                                    # nn.BatchNorm2d(d_model),
                                                )
        self.out_conv = ConvModule(
            d_model,
            in_chans2,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.out_conv2 = ConvModule(
            in_chans1,
            in_chans2,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.)


    def forward(self, x1, x2):
        # x1: input list
        # x2: output
        # patchs = []
        # pos1 = []
        # for i in range(len(x1)):
        #     patchs.append(self.to_patch1[i](x1[i]))
        k_patch = self.to_patch1(x1)
        q_patch = self.to_patch2(x2)
        b, d, h, w = q_patch.shape
        # for patch in patchs:
        #     pos1.append(self.pose_encoding(patch).transpose(0,1))
        pos1 = self.pose_encoding(k_patch).transpose(0,1)
        # pos1 = torch.zeros(h*w, b, d, dtype=tmp.dtype, device=tmp.device)
        # patchs = torch.cat([patch.flatten(2).permute(2,0,1) for patch in patchs], dim=0)
        # pos1 = torch.cat([p for p in pos1], dim=0)
        pos2 = self.pose_encoding(q_patch).transpose(0,1)
        # pos2 = torch.zeros(h*w, b, d, dtype=patchs.dtype, device=patchs.device)
        # src = x1.flatten(2).permute(2,0,1)
        # memory = self.tr_encoder(src=src, pos=pos1)
        k_patch = k_patch.flatten(2).permute(2,0,1)
        tgt = q_patch.flatten(2).permute(2,0,1)
        # out = self.tr_decoder(tgt=tgt, memory=src, pos=pos1, query_pos=pos2)
        out = self.tr_decoder(tgt=tgt, memory=k_patch, pos=pos1, query_pos=pos2)
        # out = self.tr_decoder(tgt=patchs, memory=tgt, pos=pos2, query_pos=pos1)
        out = out.permute(1,2,0).reshape(b, d, h, w)
        # out = self.out_conv(torch.cat(
        #     (x2, F.interpolate(out, size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)), dim=1))
        # out = F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + x2
        tmp = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        out = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + \
              F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        return out

@NECKS.register_module()
class Cascade_TRF3(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
        
        dim_feedforwards: [256,512,1024,2048]
        d_models: [64,128,256,512]
        patch_sizes: [[2,2]... [8,8]]
    """

    def __init__(self,
                 in_channels=[], out_channels=512, d_models=[], n_head=8,
                 dim_feedforwards=[], patch_sizes=[],
                 dropout_ratio=0.1,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=None,
                 out_type='add',
                 overlap=False,
                 cascade_num=2,
                 pos_type='sin',
                 upsample_cfg=dict(mode='nearest'),
                 adapt_size1=(10,10), 
                 adapt_size2=(10,10)):
        super(Cascade_TRF3, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_ratio = dropout_ratio
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.downdim = 512
        self.cascade_num = cascade_num
        if out_type == 'add' :
            FFTr = TransFusion5
        elif out_type == 'cat':
            FFTr = TransFusion4
        else:
            raise ValueError('{} is not surported out_type'.format(out_type))
        self.trf1s = nn.ModuleList()
        self.trf2s = nn.ModuleList()
        self.trf3s = nn.ModuleList()
        self.name_i = 0
        for _ in range(self.cascade_num):
            self.trf1s.append(FFTr(in_channels[3], in_channels[2], d_model=d_models[2], nhead=n_head, 
                                   dim_feedforward=dim_feedforwards[2], patch_size1=patch_sizes[4], patch_size2=patch_sizes[5], 
                                   norm_cfg=norm_cfg, overlap=overlap, pos_type=pos_type,adapt_size1=adapt_size1, adapt_size2=adapt_size2))
            self.trf2s.append(FFTr(in_channels[2], in_channels[1], d_model=d_models[1], nhead=n_head, 
                               dim_feedforward=dim_feedforwards[1], patch_size1=patch_sizes[2], patch_size2=patch_sizes[3], 
                               norm_cfg=norm_cfg, overlap=overlap, pos_type=pos_type,adapt_size1=adapt_size1, adapt_size2=adapt_size2))
            self.trf3s.append(FFTr(in_channels[1], in_channels[0], d_model=d_models[0], nhead=n_head, 
                                   dim_feedforward=dim_feedforwards[0], patch_size1=patch_sizes[0], patch_size2=patch_sizes[1], 
                                   norm_cfg=norm_cfg, overlap=overlap, pos_type=pos_type,adapt_size1=adapt_size1, adapt_size2=adapt_size2))

        self.backbone_end_level = self.num_outs

        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        # assert isinstance(add_extra_convs, (str, bool))
        # if isinstance(add_extra_convs, str):
            # # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            # assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        # elif add_extra_convs:  # True
            # if extra_convs_on_inputs:
                # # TODO: deprecate `extra_convs_on_inputs`
                # warnings.simplefilter('once')
                # warnings.warn(
                    # '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    # 'Please use "add_extra_convs"', DeprecationWarning)
                # self.add_extra_convs = 'on_input'
            # else:
                # self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()
        # self.downdim_conv = ConvModule(
                    # in_channels[3],
                    # self.downdim,
                    # 1,
                    # padding=0,
                    # conv_cfg=conv_cfg,
                    # norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    # act_cfg=act_cfg,
                    # inplace=False)
        for i in range(self.backbone_end_level):
            if i < 3:
                l_conv = ConvModule(
                    in_channels[i],
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            elif i == 3:
                l_conv = ConvModule(
                    in_channels[2],
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            else:
                l_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                    act_cfg=act_cfg,
                    inplace=False)
            # fpn_conv = ConvModule(
                # out_channels,
                # out_channels,
                # 3,
                # padding=1,
                # conv_cfg=conv_cfg,
                # norm_cfg=norm_cfg,
                # act_cfg=act_cfg,
                # inplace=False)

            self.lateral_convs.append(l_conv)
            # self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        # extra_levels = num_outs - self.backbone_end_level + self.start_level
        # if self.add_extra_convs and extra_levels >= 1:
            # for i in range(extra_levels):
                # if i == 0 and self.add_extra_convs == 'on_input':
                    # in_channels = self.in_channels[self.backbone_end_level - 1]
                # else:
                    # in_channels = out_channels
                # extra_fpn_conv = ConvModule(
                    # in_channels,
                    # out_channels,
                    # 3,
                    # stride=2,
                    # padding=1,
                    # conv_cfg=conv_cfg,
                    # norm_cfg=norm_cfg,
                    # act_cfg=act_cfg,
                    # inplace=False)
                # self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # outs = []
        # y4 = self.downdim_conv(inputs[3])
        # y4 = self.trf0(y4, y4)
        # y3 = self.trf1(y4, inputs[2])
        # y2 = self.trf2(y3, inputs[1])
        # y1 = self.trf3(y2, inputs[0])
        # y5 = self.lateral_convs[4](y4)
        # outs.append(self.lateral_convs[0](y1))
        # outs.append(self.lateral_convs[1](y2))
        # outs.append(self.lateral_convs[2](y3))
        # outs.append(self.lateral_convs[3](y4))
        # outs.append(y5)
        # return tuple(outs)
        outs = []
        y3 = self.trf1s[0](inputs[3], inputs[2])
        y2 = self.trf2s[0](y3, inputs[1])
        y1 = self.trf3s[0](y2, inputs[0])
        for i in range(1, self.cascade_num):
            y3 = self.trf1s[i](inputs[3], y3)
            y2 = self.trf2s[i](y3, y2)
            y1 = self.trf3s[i](y2, y1)
        y4 = self.lateral_convs[3](y3)
        outs.append(self.lateral_convs[0](y1))
        outs.append(self.lateral_convs[1](y2))
        outs.append(self.lateral_convs[2](y3))
        outs.append(y4)
        outs.append(self.lateral_convs[4](y4))
        #from tools.feature_visualization import draw_feature_map
        #draw_feature_map(tmp)
        #w = inputs[0].shape[2]
        #h = inputs[0].shape[3]
        #for i in range(4):
          #draw_feature_map(inputs[i], name=str(self.name_i)+str(i), h=h, w=w)
        #draw_feature_map(outs[4], name=str(self.name_i)+str(4), h=h, w=w)
        #self.name_i = self.name_i + 1
        return tuple(outs)
