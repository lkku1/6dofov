''' Towards An End-to-End Framework for Video Inpainting
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from model.modules.base_module import BaseNetwork
from model.modules.sparse_transformer import TemporalSparseTransformerBlock, SoftSplit, SoftComp
from model.modules.conv_pad import PadConv2d, PadConv3d

SK_Rate = 16
class DPConv(nn.Module):
    def __init__(self,in_channels,out_channels=None,dilation_list=[[1,1],[1,2],[1,4],[2,1]],stride=1,bias=False,groups=1,length=32):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.convs = nn.ModuleList()
        for dilation in dilation_list:
            self.convs.append(nn.Sequential(
                PadConv2d(in_channels,out_channels,kernel_size=3,stride=stride,dilation=dilation,bias=bias,groups=groups),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        mid_vectors = max(out_channels // SK_Rate, length)

        self.conv_weight = nn.Sequential(
            nn.Conv2d(4 * out_channels, mid_vectors, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(mid_vectors, 4 * out_channels, kernel_size=1)
        )

        self.avg2d = nn.AdaptiveAvgPool2d(1)


    def forward(self,x):

        feats = []
        for conv in self.convs:
            feats.append(conv(x))
        feats = torch.cat(feats, dim=1)  # [B, N * C, H, W]
        b, nc, h, w = feats.size()

        vector = self.avg2d(feats)

        conv_weight = torch.softmax(self.conv_weight(vector).view(b, 4, nc//4, 1, 1), dim=1)
        feats = torch.sum(feats.view(b, 4, nc//4, h, w) * conv_weight, dim=1)

        return feats

class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]


        self.conv1 = nn.Sequential(PadConv2d(in_channel, 64, kernel_size=3, stride=2), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = DPConv(64, 64, stride=1)
        self.conv3 = DPConv(64, 128, stride=2)
        self.conv4 = DPConv(128, 256, stride=1)

        self.conv5 = DPConv(256, 384, stride=1)
        self.conv6 = DPConv(640, 512, stride=1, groups=2)
        self.conv7 = DPConv(768, 384, stride=1, groups=4)
        self.conv8 = DPConv(640, 256, stride=1, groups=8)
        self.conv9 = DPConv(512, 128, stride=1)


    def forward(self, x):
        bt, c, _, _ = x.size()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x0 = self.conv4(x)

        _, _, h, w = x.size()
        x = self.conv5(x0)

        x = self.conv6(torch.cat([x.view(bt, 2, -1, h, w), x0.view(bt, 2, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv7(torch.cat([x.view(bt, 4, -1, h, w), x0.view(bt, 4, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv8(torch.cat([x.view(bt, 8, -1, h, w), x0.view(bt, 8, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv9(torch.cat([x.view(bt, 1, -1, h, w), x0.view(bt, 1, -1, h, w)], dim=2).view(bt, -1, h, w))

        return x

class Condition_Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Condition_Encoder, self).__init__()

        self.conv1 = nn.Sequential(PadConv2d(in_channels, 64, kernel_size=3, stride=2), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = DPConv(64, 64, stride=1)

        self.conv3 = DPConv(64, 96, stride=2)
        self.conv4 = DPConv(96, 128, stride=1)

        self.conv5 = DPConv(128, 192, stride=1)
        self.conv6 = DPConv(320, 168, stride=1, groups=2)
        self.conv7 = DPConv(296, 144, stride=1, groups=4)
        self.conv8 = DPConv(272, 120, stride=1, groups=8)
        self.conv9 = DPConv(248, 128, stride=1)

    def forward(self, x):
        bt, c, _, _ = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x0 = self.conv4(x)

        _, _, h, w = x.size()
        x = self.conv5(x0)

        x = self.conv6(torch.cat([x.view(bt, 2, -1, h, w), x0.view(bt, 2, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv7(torch.cat([x.view(bt, 4, -1, h, w), x0.view(bt, 4, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv8(torch.cat([x.view(bt, 8, -1, h, w), x0.view(bt, 8, -1, h, w)], dim=2).view(bt, -1, h, w))
        x = self.conv9(torch.cat([x.view(bt, 1, -1, h, w), x0.view(bt, 1, -1, h, w)], dim=2).view(bt, -1, h, w))

        return x


class deconv(nn.Module):
    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 ):
        super().__init__()
        self.conv = PadConv2d(input_channel,
                              output_channel,
                              kernel_size=kernel_size,
                              stride=1,
                             )

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, model_path=None, in_channel=5, out_channel=1):
        super(InpaintGenerator, self).__init__()
        channel = 128
        hidden = 512

        # encoder
        self.encoder = Encoder(in_channel)
        self.condition_encoder = Condition_Encoder(in_channels=4)
        self.out_channel = out_channel

        # decoder
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            PadConv2d(128, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3),
            nn.LeakyReLU(0.2, inplace=True),
            PadConv2d(64, out_channel, kernel_size=3, stride=1),
            )

        # soft split and soft composition
        kernel_size = (7, 7)
        stride = (3, 3)
        t2t_params = {
            'kernel_size': kernel_size,
            'stride': stride,
        }
        self.ss = SoftSplit(channel, hidden, kernel_size, stride)
        self.sc = SoftComp(channel, hidden, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(kernel_size, stride)

        depths = 8
        num_heads = 4
        window_size = (5, 9)
        pool_size = (4, 4)
        self.transformers = TemporalSparseTransformerBlock(dim=hidden,
                                                n_head=num_heads,
                                                window_size=window_size,
                                                pool_size=pool_size,
                                                depths=depths,
                                                t2t_params=t2t_params)
        if init_weights:
            self.init_weights()

        if model_path is not None:
            print('Pretrained ProPainter has loaded...')
            ckpt = torch.load(model_path, map_location='cpu')
            self.load_state_dict(ckpt, strict=True)

        # print network parameter number
        self.print_network()


    def forward(self, masked_frames, completed_flows, masks_in, mode="constant"):
        """
        Args:
            masks_in: original mask
            masks_updated: updated mask after image propagation
        """
        b, t, ori_c, ori_h, ori_w = masked_frames.size()
        masked_frames = masked_frames.view(b*t, ori_c, ori_h, ori_w)
        completed_flows = completed_flows.view(b*t, -1, ori_h, ori_w)

        # extracting features
        enc_feat = self.encoder(masked_frames)
        _, c, h, w = enc_feat.size()
        fold_feat_size = torch.tensor((h, w))

        enc_condition = self.condition_encoder(completed_flows)

        trans_feat, pad_feat_size, pad_size = self.ss(enc_feat, b, fold_feat_size, mode)
        enc_condition, _, _ = self.ss(enc_condition, b, fold_feat_size, mode)

        ds_mask_in = F.interpolate(masks_in.reshape(-1, 1, ori_h, ori_w), scale_factor=1 / 4, mode='nearest')
        ds_mask_in = F.pad(F.pad(ds_mask_in, pad=(pad_size[2], pad_size[3], 0, 0), mode=mode), pad=(0, 0, pad_size[0], pad_size[1]))
        ds_mask_in = self.max_pool(ds_mask_in)
        ds_mask_in = ds_mask_in.view(b, t, 1, ds_mask_in.shape[2], ds_mask_in.shape[3])
        mask_pool_l = rearrange(ds_mask_in, 'b t c h w -> b t h w c').contiguous()

        trans_feat = self.transformers(trans_feat, enc_condition, fold_feat_size, mask_pool_l)
        trans_feat = self.sc(trans_feat, t, pad_feat_size)
        trans_feat = trans_feat[:, :, pad_size[0]:-pad_size[1], pad_size[2]:-pad_size[3]]
        trans_feat = trans_feat.view(b, t, -1, h, w)

        enc_feat = enc_feat + trans_feat

        output = self.decoder(enc_feat.view(-1, c, h, w))
        output = output.view(b, t, self.out_channel, ori_h, ori_w)

        return output


# ######################################################################
#  Discriminator for Temporal Patch GAN
# ######################################################################
class Discriminator(BaseNetwork):
    def __init__(self,
                 in_channels=1,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(

            PadConv3d(in_channels, nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            PadConv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            PadConv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            PadConv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            PadConv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2), bias=not use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),

            PadConv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2)))

        if init_weights:
            self.init_weights()


    def forward(self, xs):
        # T, C, H, W = xs.shape (old)
        # B, T, C, H, W (new)
        xs_t = torch.transpose(xs, 1, 2).contiguous()
        feat = self.conv(xs_t)

        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2).contiguous() # B, T, C, H, W
        return out





if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms
    a = Image.open('/media/lyb/CE7258D87258C73D/linux/github2/dynamic_panorama/480p/scene1/00000.jpg')
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    blur_tensor = transform(a)
    a = Discriminator(in_channels=1).cuda()
    b = torch.ones((1, 5, 1, 240, 432)).cuda()
    c = a(b)
    print("a")






