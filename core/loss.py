import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """
    def __init__(self,
                 type='nsgan',
                 target_real_label=1.0,
                 target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None, ):

        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return (self.criterion(1 + outputs)).mean()
            else:
                return ((-outputs)).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class CalculateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.vgg_extract = VGG16Extractor()
        # self.l1 = nn.L1Loss()
        self.vgg_loss = VGG16PartialLoss(perceptual_alpha=1, style_alpha=1, smooth_alpha=1)

    @staticmethod
    def normalize_batch(batch, div_factor=255.):
        """
        Normalize batch

        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        mean[:, 3, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        std[:, 3, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch


    def forward(self, composed, output, con_mask, occ_mask, ground_truth, config):
        b, t, c, h, w = composed.size()
        composed = composed.view(-1, c, h, w)
        output = output.view(-1, c, h, w)
        ground_truth = ground_truth.view(-1, c, h, w)
        con_mask = con_mask.view(-1, 1, h, w)
        occ_mask = occ_mask.view(-1, 1, h, w)

        composed = self.normalize_batch(composed, 1)
        output = self.normalize_batch(output, 1)
        ground_truth = self.normalize_batch(ground_truth, 1)

        perceptual, style, smooth = self.vgg_loss(composed[:, :-1,...], ground_truth[:, :-1,...])


        loss_dict = dict()
        loss_dict["hole"] = torch.abs(occ_mask * composed - occ_mask * ground_truth).mean() * config['losses']['hole_weight']
        loss_dict["valid"] = torch.abs(con_mask * composed - con_mask * ground_truth).mean() * config['losses']['valid_weight']
        loss_dict["perceptual"] = perceptual * config['losses']['perceptual_weight']
        loss_dict["style"] = style * config['losses']['style_weight']
        loss_dict["tv"] = smooth * config['losses']['vt_weight']
        return loss_dict


def gram_matrix(input_tensor):
    """
    Compute Gram matrix

    :param input_tensor: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :return: Gram matrix of y
    """
    (b, ch, h, w) = input_tensor.size()
    features = input_tensor.view(b, ch, w * h)
    features_t = features.transpose(1, 2)

    # more efficient and formal way to avoid underflow for mixed precision training
    input = torch.zeros(b, ch, ch).type(features.type())
    gram = torch.baddbmm(input, features, features_t, beta=0, alpha=1. / (ch * h * w), out=None)

    # naive way to avoid underflow for mixed precision training
    # features = features / (ch * h)
    # gram = features.bmm(features_t) / w

    # for fp32 training, it is also safe to use the following:
    # gram = features.bmm(features_t) / (ch * h * w)

    return gram


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss Module
    """

    def __init__(self):
        """Init"""
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.mse_loss = torch.nn.MSELoss()

    @staticmethod
    def normalize_batch(batch, div_factor=1.):
        """
        Normalize batch

        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def forward(self, x, y):
        """
        Forward

        :param x: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param y: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: l1 loss between the normalized data
        """
        x = self.normalize_batch(x)
        y = self.normalize_batch(y)
        return self.l1_loss(x, y)


def make_vgg16_layers(style_avg_pool=False):
    """
    make_vgg16_layers

    Return a custom vgg16 feature module with avg pooling
    """
    vgg16_cfg = [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
        512, 512, 512, 'M', 512, 512, 512, 'M'
    ]

    layers = []
    in_channels = 3
    for v in vgg16_cfg:
        if v == 'M':
            if style_avg_pool:
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Partial(nn.Module):
    """
    VGG16 partial model
    """

    def __init__(self, vgg_path='~/.torch/vgg16-397923af.pth', layer_num=3):
        """
        Init

        :param layer_num: number of layers
        """
        super().__init__()
        vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_model.features = make_vgg16_layers()
        # vgg_model.load_state_dict(
        #     torch.load(vgg_path, map_location='cpu')
        # )
        vgg_pretrained_features = vgg_model.features

        assert layer_num > 0
        assert isinstance(layer_num, int)
        self.layer_num = layer_num

        self.slice1 = torch.nn.Sequential()
        for x in range(5):  # 4
            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 1:
            self.slice2 = torch.nn.Sequential()
            for x in range(5, 10):  # (4, 9)
                self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 2:
            self.slice3 = torch.nn.Sequential()
            for x in range(10, 17):  # (9, 16)
                self.slice3.add_module(str(x), vgg_pretrained_features[x])

        if self.layer_num > 3:
            self.slice4 = torch.nn.Sequential()
            for x in range(17, 24):  # (16, 23)
                self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    @staticmethod
    def normalize_batch(batch, div_factor=1.0):
        """
        Normalize batch

        :param batch: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :param div_factor: normalizing factor before data whitening
        :return: normalized data, tensor with shape
         (batch_size, nbr_channels, height, width)
        """
        # normalize using imagenet mean and std
        mean = batch.data.new(batch.data.size())
        std = batch.data.new(batch.data.size())
        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406
        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225
        batch = torch.div(batch, div_factor)

        batch -= Variable(mean)
        batch = torch.div(batch, Variable(std))
        return batch

    def forward(self, x):
        """
        Forward, get features used for perceptual loss

        :param x: input tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: list of self.layer_num feature maps used to compute the
         perceptual loss
        """
        h = self.slice1(x)
        h1 = h

        output = []

        if self.layer_num == 1:
            output = [h1]
        elif self.layer_num == 2:
            h = self.slice2(h)
            h2 = h
            output = [h1, h2]
        elif self.layer_num == 3:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            output = [h1, h2, h3]
        elif self.layer_num >= 4:
            h = self.slice2(h)
            h2 = h
            h = self.slice3(h)
            h3 = h
            h = self.slice4(h)
            h4 = h
            output = [h1, h2, h3, h4]
        return output


# perceptual loss and (spatial) style loss
class VGG16PartialLoss(PerceptualLoss):
    """
    VGG16 perceptual loss
    """

    def __init__(self, l1_alpha=5.0, perceptual_alpha=0.05, style_alpha=120.0,
                 smooth_alpha=0.0, feat_num=3, vgg_path='~/.torch/vgg16-397923af.pth'):
        """
        Init

        :param l1_alpha: weight of the l1 loss
        :param perceptual_alpha: weight of the perceptual loss
        :param style_alpha: weight of the style loss
        :param smooth_alpha: weight of the regularizer
        :param feat_num: number of feature maps
        """
        super().__init__()

        self.vgg16partial = VGG16Partial().eval()

        self.loss_fn = torch.nn.L1Loss(reduction='mean')

        self.l1_weight = l1_alpha
        self.vgg_weight = perceptual_alpha
        self.style_weight = style_alpha
        self.regularize_weight = smooth_alpha

        self.dividor = 1
        self.feat_num = feat_num

    def forward(self, output0, target0):
        """
        Forward

        assuming both output0 and target0 are in the range of [0, 1]

        :param output0: output of a model, tensor with shape
         (batch_size, nbr_channels, height, width)
        :param target0: target, tensor with shape
         (batch_size, nbr_channels, height, width)
        :return: total perceptual loss
        """
        # y = self.normalize_batch(target0, self.dividor)
        # x = self.normalize_batch(output0, self.dividor)

        y = target0
        x = output0

        # L1 loss
        # l1_loss = self.l1_weight * (torch.abs(x - y).mean())
        vgg_loss = 0
        style_loss = 0
        smooth_loss = 0

        # VGG
        if self.vgg_weight != 0 or self.style_weight != 0:

            yc = Variable(y.data)

            with torch.no_grad():
                groundtruth = self.vgg16partial(yc)
            generated = self.vgg16partial(x)

            # vgg loss: VGG content loss
            if self.vgg_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(len(generated) - self.feat_num, len(generated)):
                    gt_data = Variable(groundtruth[m].data, requires_grad=False)
                    vgg_loss += (
                            self.vgg_weight * self.loss_fn(generated[m], gt_data)
                    )

            # style loss: Gram matrix loss
            if self.style_weight > 0:
                # for m in range(0, len(generated)):
                for m in range(len(generated) - self.feat_num, len(generated)):
                    gt_style = gram_matrix(
                        Variable(groundtruth[m].data, requires_grad=False))
                    gen_style = gram_matrix(generated[m])
                    style_loss += (
                            self.style_weight * self.loss_fn(gen_style, gt_style)
                    )

        # smooth term
        if self.regularize_weight != 0:
            smooth_loss += self.regularize_weight * (
                    torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean() +
                    torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()
            )

        # tot = l1_loss + vgg_loss + style_loss + smooth_loss
        return vgg_loss, style_loss, smooth_loss