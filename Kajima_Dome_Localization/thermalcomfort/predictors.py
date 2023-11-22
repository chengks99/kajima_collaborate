import torch.nn.functional as F
import torch
import numpy as np
from checkpoint import load_checkpoint
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn
import builder

# yolo testing

from registry import (ATTRPREDICTOR, BACKBONES, CATEPREDICTOR, CONCATS,
                             GLOBALPOOLING, LOSSES, PREDICTOR, ROIPOOLING)


class BasePredictor(nn.Module):
    ''' Base class for attribute predictors'''
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BasePredictor, self).__init__()

    @property
    def with_roi_pool(self):
        return hasattr(self, 'roi_pool') and self.roi_pool is not None

    @abstractmethod
    def simple_test(self, img, landmark):
        pass

    @abstractmethod
    def aug_test(self, img, landmark):
        pass

    def forward_test(self, img, landmark=None):
        num_augs = len(img)
        if num_augs == 1:  # single image test
            return self.simple_test(img[0], landmark[0])
        else:
            return self.aug_test(img, landmark)

    @abstractmethod
    def forward_train(self, img, landmark, attr, cate):
        pass

    def forward(self,
                img,
                attr=None,
                cate=None,
                landmark=None,
                return_loss=True):
        if return_loss:
            return self.forward_train(img, landmark, attr, cate)
        else:
            return self.forward_test(img, landmark)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))


@PREDICTOR.register_module
class RoIAttrCatePredictor(BasePredictor):

    def __init__(self,
                 backbone,
                 global_pool,
                 roi_pool,
                 concat,
                 attr_predictor,
                 cate_predictor,
                 pretrained=None):
        super(RoIAttrCatePredictor, self).__init__()

        self.backbone = builder.build_backbone(backbone)
        self.global_pool = builder.build_global_pool(global_pool)

        assert roi_pool is not None
        self.roi_pool = builder.build_roi_pool(roi_pool)

        self.concat = builder.build_concat(concat)
        self.attr_predictor = builder.build_attr_predictor(attr_predictor)
        self.cate_predictor = builder.build_cate_predictor(cate_predictor)

        self.init_weights(pretrained)

    def forward_train(self, x, landmark, attr, cate):
        # 1. conv layers extract global features
        x = self.backbone(x)

        # 2. global pooling
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)

        # 3. roi pooling
        local_x = self.roi_pool(x, landmark)

        # 4. concat
        feat = self.concat(global_x, local_x)

        # 5. attribute prediction
        losses = dict()
        losses['loss_attr'] = self.attr_predictor(feat, attr, return_loss=True)
        losses['loss_cate'] = self.cate_predictor(feat, cate, return_loss=True)

        return losses

    def simple_test(self, x, landmark=None):
        """Test single image"""
        x = x.unsqueeze(0)
        landmark = landmark.unsqueeze(0)
        attr_pred, cate_pred = self.aug_test(x, landmark)
        return attr_pred[0], cate_pred[0]

    def aug_test(self, x, landmark=None):
        """Test batch of images"""
        x = self.backbone(x)
        global_x = self.global_pool(x)
        global_x = global_x.view(global_x.size(0), -1)
        local_x = self.roi_pool(x, landmark)

        feat = self.concat(global_x, local_x)
        attr_pred = self.attr_predictor(feat)
        cate_pred = self.cate_predictor(feat)
        return attr_pred, cate_pred

    def init_weights(self, pretrained=None):
        self.backbone.init_weights(pretrained=pretrained)
        self.global_pool.init_weights()
        self.roi_pool.init_weights()
        self.concat.init_weights()
        self.attr_predictor.init_weights()
        self.cate_predictor.init_weights()


# 33


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers
        # downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BACKBONES.register_module
class ResNet(nn.Module):
    layer_setting = {
        'resnet50': [3, 4, 6, 3],
        'resnet18': [2, 2, 2, 2],
        'resnet34': [3, 4, 6, 3]
    }

    block_setting = {
        'resnet18': BasicBlock,
        'resnet34': BasicBlock,
        'resnet50': Bottleneck
    }

    def __init__(self,
                 setting='resnet50',
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):

        super(ResNet, self).__init__()
        block = self.block_setting[setting]
        layers = self.layer_setting[setting]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])

        self.zero_init_residual = zero_init_residual

    def init_weights(self, pretrained=None):
        print('pretrained model', pretrained)
        if isinstance(pretrained, str):
            load_checkpoint(self, pretrained)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch, so that the
            # residual branch starts with zeros, and each residual block
            # behaves like an identity. This improves the model by 0.2~0.3%
            # according to https://arxiv.org/abs/1706.02677
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


#######################################
@GLOBALPOOLING.register_module
class GlobalPooling(nn.Module):

    def __init__(self, inplanes, pool_plane, inter_channels, outchannels):
        super(GlobalPooling, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(inplanes)

        inter_plane = inter_channels[0] * inplanes[0] * inplanes[1]
        if len(inter_channels) > 1:
            self.global_layers = nn.Sequential(
                nn.Linear(inter_plane, inter_channels[1]),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(inter_channels[1], outchannels),
                nn.ReLU(True),
                nn.Dropout(),
            )
        else:  # just one linear layer
            self.global_layers = nn.Linear(inter_plane, outchannels)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        global_pool = self.global_layers(x)
        return global_pool

    def init_weights(self):
        if isinstance(self.global_layers, nn.Linear):
            nn.init.normal_(self.global_layers.weight, 0, 0.01)
            if self.global_layers.bias is not None:
                nn.init.constant_(self.global_layers.bias, 0)
        elif isinstance(self.global_layers, nn.Sequential):
            for m in self.global_layers:
                if type(m) == nn.Linear:
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


###########################


@ROIPOOLING.register_module
class RoIPooling(nn.Module):

    def __init__(self,
                 pool_plane,
                 inter_channels,
                 outchannels,
                 crop_size=7,
                 img_size=(224, 224),
                 num_lms=8,
                 roi_size=2):
        super(RoIPooling, self).__init__()
        self.maxpool = nn.MaxPool2d(pool_plane)
        self.linear = nn.Sequential(
            nn.Linear(num_lms * inter_channels, outchannels), nn.ReLU(True),
            nn.Dropout())

        self.inter_channels = inter_channels
        self.outchannels = outchannels
        self.num_lms = num_lms
        self.crop_size = crop_size
        assert img_size[0] == img_size[
            1], 'img width should equal to img height'
        self.img_size = img_size[0]
        self.roi_size = roi_size

        self.a = self.roi_size / float(self.crop_size)
        self.b = self.roi_size / float(self.crop_size)

    def forward(self, features, landmarks):
        """batch-wise RoI pooling.

        Args:
            features(tensor): the feature maps to be pooled.
            landmarks(tensor): crop the region of interest based on the
                landmarks(bs, self.num_lms).
        """
        batch_size = features.size(0)

        # transfer landmark coordinates from original image to feature map
        landmarks = landmarks / self.img_size * self.crop_size

        landmarks = landmarks.view(batch_size, self.num_lms, 2)

        ab = [np.array([[self.a, 0], [0, self.b]]) for _ in range(batch_size)]
        ab = np.stack(ab, axis=0)
        ab = torch.from_numpy(ab).float().cuda()
        size = torch.Size(
            (batch_size, features.size(1), self.roi_size, self.roi_size))

        pooled = []
        for i in range(self.num_lms):
            tx = -1 + 2 * landmarks[:, i, 0] / float(self.crop_size)
            ty = -1 + 2 * landmarks[:, i, 1] / float(self.crop_size)
            t_xy = torch.stack((tx, ty)).view(batch_size, 2, 1)
            theta = torch.cat((ab, t_xy), 2)

            flowfield = nn.functional.affine_grid(theta, size, align_corners = True)
            one_pooled = nn.functional.grid_sample(
                features,
                flowfield.to(torch.float32),
                mode='bilinear',
                padding_mode='border', align_corners = True)
            one_pooled = self.maxpool(one_pooled).view(batch_size,
                                                       self.inter_channels)

            pooled.append(one_pooled)
        pooled = torch.stack(pooled, dim=1).view(batch_size, -1)
        pooled = self.linear(pooled)
        return pooled

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


#####################################

@CONCATS.register_module
class Concat(nn.Module):

    def __init__(self, inchannels, outchannels):
        super(Concat, self).__init__()
        # concat global and local
        self.fc_fusion = nn.Linear(inchannels, outchannels)

    def forward(self, global_x, local_x=None):
        if local_x is not None:
            x = torch.cat((global_x, local_x), 1)
            x = self.fc_fusion(x)
        else:
            x = global_x

        return x

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_fusion.weight)
        if self.fc_fusion.bias is not None:
            self.fc_fusion.bias.data.fill_(0.01)

########################################


@ATTRPREDICTOR.register_module
class AttrPredictor(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 loss_attr=dict(
                     type='BCEWithLogitsLoss',
                     ratio=1,
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean')):
        super(AttrPredictor, self).__init__()
        self.linear_attr = nn.Linear(inchannels, outchannels)
        self.loss_attr = builder.build_loss(loss_attr)

    def forward_train(self, x, attr):
        attr_pred = self.linear_attr(x)
        loss_attr = self.loss_attr(attr_pred, attr)
        return loss_attr

    def forward_test(self, x):
        attr_pred = torch.sigmoid(self.linear_attr(x))
        return attr_pred

    def forward(self, x, attr=None, return_loss=False):
        if return_loss:
            return self.forward_train(x, attr)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_attr.weight)
        if self.linear_attr.bias is not None:
            self.linear_attr.bias.data.fill_(0.01)


#######################################


@CATEPREDICTOR.register_module
class CatePredictor(nn.Module):

    def __init__(self,
                 inchannels,
                 outchannels,
                 loss_cate=dict(
                     type='CELoss',
                     ratio=1,
                     weight=None,
                     size_average=None,
                     reduce=None,
                     reduction='mean')):
        super(CatePredictor, self).__init__()
        self.linear_cate = nn.Linear(inchannels, outchannels)
        self.loss_cate = builder.build_loss(loss_cate)

    def forward_train(self, x, cate):
        cate_pred = self.linear_cate(x)
        loss_cate = self.loss_cate(cate_pred, cate)
        return loss_cate

    def forward_test(self, x):
        cate_pred = torch.sigmoid(self.linear_cate(x))
        return cate_pred

    def forward(self, x, cate=None, return_loss=False):
        if return_loss:
            return self.forward_train(x, cate)
        else:
            return self.forward_test(x)

    def init_weights(self):
        nn.init.xavier_uniform_(self.linear_cate.weight)
        if self.linear_cate.bias is not None:
            self.linear_cate.bias.data.fill_(0.01)


##############################################


@LOSSES.register_module
class BCEWithLogitsLoss(nn.Module):

    def __init__(self, ratio, weight, size_average, reduce, reduction):
        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduce = reduce
        self.reduction = reduction

        self.ratio = ratio

    def forward(self, input, target):
        target = target.float()
        return self.ratio * F.binary_cross_entropy_with_logits(
            input, target, self.weight, reduction=self.reduction)


################################################

@LOSSES.register_module
class CELoss(nn.Module):

    def __init__(self,
                 ratio=1,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        super(CELoss, self).__init__()
        self.ratio = ratio
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        Calculate the cross-entropy loss
        :param input(torch.Tensor): The prediction with shape (N, C),
                                    C is the number of classes.
        :param target(torch.Tensor): The learning label(N, 1) of
                                     the prediction.
        :return: (torch.Tensor): The calculated loss
        """
        target = target.squeeze_()
        return self.ratio * F.cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction)
