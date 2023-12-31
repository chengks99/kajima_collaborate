from __future__ import absolute_import

import mxnet as mx
import cv2
import logging
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import normalize

import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as T

import osnet_ain as oa

class FaceEmbedding(object):

    def __init__(self, model_path, model_epoch, device, input_size, emb_layer_name="fc1_output"):
        print("Loading Feature Extraction Model...")

        self.mx_device = mx.cpu(0)
        if device >= 0:
            self.mx_device = mx.gpu(device)

        self.Batch = namedtuple('Batch', ['data'])
        # loading model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        all_layers = sym.get_internals()
        fe_sym = all_layers[emb_layer_name]  # 'fc1_output' by default

        self.fe_mod = mx.mod.Module(symbol=fe_sym, context=self.mx_device, label_names=None)
        self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, input_size, input_size))])
        self.fe_mod.set_params(arg_params, aux_params)

        print("Feature Extraction Loaded.")

    def extract_feature(self, input_image, normalise_features=False):
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        self.fe_mod.forward(self.Batch([mx.nd.array(img)]))
        feature = self.fe_mod.get_outputs()[0].asnumpy()

        if not normalise_features:
            return feature

        feature_norm = normalize(feature)
        return feature_norm

    def extract_feature_batch(self, img_list):
        images = np.swapaxes(img_list, 1, 3)
        images = np.swapaxes(images, 2, 3)

        self.fe_mod.forward(self.Batch([mx.nd.array(images)]))
        features = self.fe_mod.get_outputs()[0].asnumpy()

        return features

class UbodyEmbedding(object):

    def __init__(self, model_path, model_epoch, device, input_size, emb_layer_name="fc1_output"):

        self.mx_device = mx.cpu(0)
        if device >= 0:
            self.mx_device = mx.gpu(device)

        self.Batch = namedtuple('Batch', ['data'])
        # loading model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        all_layers = sym.get_internals()
        fe_sym = all_layers[emb_layer_name]  # 'fc1_output' by default

        self.fe_mod = mx.mod.Module(symbol=fe_sym, context=self.mx_device, label_names=None)
        self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, input_size, input_size))])
        self.fe_mod.set_params(arg_params, aux_params)

        # dummy_data = [mx.nd.ones((1, 3, input_size, input_size))]
        # self.fe_mod.forward(self.Batch(dummy_data))
        # dummy_output = self.fe_mod.get_outputs()[0].asnumpy()
        # del dummy_data, dummy_output

        logging.debug('UbodyEmbedding loaded ...')

    def extract_feature(self, input_image, normalise_features=False):
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        self.fe_mod.forward(self.Batch([mx.nd.array(img)]))
        feature = self.fe_mod.get_outputs()[0].asnumpy()

        if not normalise_features:
            return feature

        feature_norm = normalize(feature)
        return feature_norm

    def extract_feature_batch(self, img_list):
        images = np.swapaxes(img_list, 1, 3)
        images = np.swapaxes(images, 2, 3)

        self.fe_mod.forward(self.Batch([mx.nd.array(images)]))
        try:
            features = self.fe_mod.get_outputs()[0].asnumpy()
        except ValueError:
            features = None

        return features

__model_factory = {
    'osnet_ain_x1_0': oa.osnet_ain_x1_0
}

def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(
    name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )

def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


class FullbodyEmbedding(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device=-1,
        verbose=True
    ):
        if device<0:
            self.device = torch.device("cpu")
            use_gpu = False
        else: 
            self.device = torch.device("cuda:{}".format(device))
            use_gpu = True

        # Build model
        model = build_model(
            model_name,
            num_classes=1,
            pretrained=False,
            use_gpu=use_gpu
        )
        model.eval()

        if verbose:
            #num_params, flops = compute_model_complexity(
            #    model, (1, 3, image_size[0], image_size[1])
            #)
            print('Model: {}'.format(model_name))

        if model_path and check_isfile(model_path):
            load_pretrained_weights(model, model_path)

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        model.to(device)

        # Class attributes
        self.model = model
        self.preprocess = preprocess
        self.to_pil = to_pil

    def __call__(self, input):
        images = []
        for element in input:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        images = images.to(self.device)

        with torch.no_grad():
            features = self.model(images)

        return features.cpu()
