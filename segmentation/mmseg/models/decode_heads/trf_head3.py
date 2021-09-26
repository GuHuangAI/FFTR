import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
import math
from mmseg.ops import resize
from ..builder import HEADS
from mmcv.cnn import normal_init
from mmcv.runner import auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy

from abc import ABCMeta, abstractmethod

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

class TransFusion4(nn.Module):
    # concat for output
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
            self.pose_encoding = PositionEmbeddingSine(normalize=True)  ### surport soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()
        else:
            raise ValueError('unsurported pos_type')
        # self.patch_size = patch_size
        if overlap:
            stride_size1 = (patch_size1[0]//2, patch_size1[1]//2)
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
            d_model+in_chans1,
            in_chans2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'))
        # self.out_conv2 = ConvModule(
        #     in_chans1,
        #     in_chans2,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     conv_cfg=None,
        #     norm_cfg=norm_cfg,
        #     act_cfg=dict(type='ReLU'))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
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

class TransFusion5(nn.Module):
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin'):
        super(TransFusion5, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerDecoderLayer2(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if pos_type == 'sin':
            self.pose_encoding = PositionEmbeddingSine(normalize=True)
        elif pos_type == 'learn':
            self.pose_encoding = PositionEmbeddingSine(normalize=True) ### surport soon
        elif pos_type == 'zero':
            self.pose_encoding = PositionEmbeddingZero()
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
        out = F.interpolate(self.out_conv2(x1), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True) + \
              F.interpolate(self.out_conv(out), size=(x2.shape[-2], x2.shape[-1]), mode='bilinear', align_corners=True)
        return out

@HEADS.register_module()
class TRFHead3(nn.Module):
    """
    dim_feedforwards: [256,512,1024]
    d_models: [64,128,256]
    patch_sizes: [[2,2]... [8,8]]
    """

    def __init__(self, in_channels=[], channels=512, d_models=[], n_head=8,
                 dim_feedforwards=[], patch_sizes=[], num_classes=19,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 out_type='add',
                 overlap='False',
                 last_conv_dilated=False,
                 **kwargs):
        super(TRFHead3, self).__init__(**kwargs)
        assert len(dim_feedforwards) == len(d_models)
        assert out_type in ['add', 'cat']
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None
        # assert isinstance(dilations, (list, tuple))
        # self.dilations = dilations
        self.in_channels = in_channels  #[256,512,1024,2048] for r50 and r101
        self.channels = channels
        self.num_classes = num_classes
        if out_type == 'add' :
            FFTr = TransFusion5
        elif out_type == 'cat':
            FFTr = TransFusion4
        else:
            raise ValueError('{} is not surported out_type'.format(out_type))
        self.trf1 = FFTr(in_channels[3], in_channels[2], d_model=d_models[2], nhead=n_head,
                         dim_feedforward=dim_feedforwards[2], patch_size1=patch_sizes[4], patch_size2=patch_sizes[5],
                         norm_cfg=self.norm_cfg, overlap=overlap)
        self.trf2 = FFTr(in_channels[2], in_channels[1], d_model=d_models[1], nhead=n_head,
                         dim_feedforward=dim_feedforwards[1], patch_size1=patch_sizes[2], patch_size2=patch_sizes[3],
                         norm_cfg=self.norm_cfg, overlap=overlap)
        self.trf3 = FFTr(in_channels[1], in_channels[0], d_model=d_models[0], nhead=n_head,
                         dim_feedforward=dim_feedforwards[0], patch_size1=patch_sizes[0], patch_size2=patch_sizes[1],
                         norm_cfg=self.norm_cfg, overlap=overlap)
        # self.bottleneck = nn.Sequential(
        #                 nn.ReLU(),
        #                 # SPHead(512, 128, bias=False),
        #                 nn.Conv2d(2*in_channels[0]+in_channels[1]+in_channels[2], channels, kernel_size=3, padding=1, stride=0),
        #                 nn.BatchNorm2d(channels),
        #                 nn.ReLU()
        #                 )
        if not last_conv_dilated:
            self.bottleneck = ConvModule(
                len(in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.bottleneck = ConvModule(
                len(in_channels) * self.channels,
                self.channels,
                3,
                padding=2,
                dilation=2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        nn.init.normal_(self.conv_seg.weight, 0, 0.01)
        nn.init.constant_(self.conv_seg.bias, 0)

    def init_weights(self):
        """Initialize weights of classification layer."""
        # normal_init(self.conv_seg, mean=0, std=0.01)
        pass

    def forward(self, inputs):
        """Forward function."""
        size = inputs[0].shape[2:]
        y1 = self.trf1(inputs[3], inputs[2])
        y2 = self.trf2(y1, inputs[1])
        y3 = self.trf3(y2, inputs[0])
        output = torch.cat((inputs[0],
                            F.interpolate(y3, size=size, mode='bilinear', align_corners=True),
                            F.interpolate(y2, size=size, mode='bilinear', align_corners=True),
                            F.interpolate(y1, size=size, mode='bilinear', align_corners=True)), dim=1)
        output = self.bottleneck(output)
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.conv_seg(output)
        return output

    @auto_fp16()
    @abstractmethod
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        loss = dict()
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)
        loss['loss_seg'] = self.loss_decode(
            seg_logit,
            seg_label,
            weight=seg_weight,
            ignore_index=self.ignore_index)
        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss
