import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def forward_pre(self, tgt,
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

class TransFusion5(nn.Module):
    def __init__(self, in_chans1, in_chans2, d_model=128, nhead=8, dim_feedforward=512, patch_size1=(12, 12),
                 patch_size2=(12, 12), dropout=0.1, activation='relu', normalize_before=False,
                 norm_cfg=dict(type='BN', requires_grad=True), overlap=False, pos_type='sin'):
        super(TransFusion5, self).__init__()
        # self.tr_encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        self.tr_decoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
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

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class MSCA_PSPHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = self.cls_seg(output)
        return output
