from __future__ import division
import inspect

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


def build_from_cfg(cfg, registry, default_args=None):
    """ build a module from config dict
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        obj_type = registry.get(obj_type)
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


BACKBONES = Registry('backbone')
GLOBALPOOLING = Registry('global_pool')  # global pooling
ROIPOOLING = Registry('roi_pool')  # roi pooling
CONCATS = Registry('concat')  # concat local features and global features
ATTRPREDICTOR = Registry('attr_predictor')  # predict attributes
CATEPREDICTOR = Registry('cate_predictor')  # predict category
EMBEDEXTRACTOR = Registry('embed_extractor')  # extract embeddings

LANDMARKFEATUREEXTRACTOR = Registry('landmark_feature_extractor')
VISIBILITYCLASSIFIER = Registry('visibility_classifier')
LANDMARKREGRESSION = Registry('landmark_regression')

LOSSES = Registry('loss')  # loss function

PREDICTOR = Registry('predictor')

RETRIEVER = Registry('retriever')

LANDMARKDETECTOR = Registry('landmark_detector')

TYPESPECIFICNET = Registry('type_specific_net')
TRIPLETNET = Registry('triplet_net')

RECOMMENDER = Registry('fashion_recommender')

FEATUREEXTRACTOR = Registry('feature_extractor')
FEATURENORM = Registry('feature_norm')
FEATURECORRELATION = Registry('feature_correlation')
FEATUREREGRESSION = Registry('feature_regression')

TPSWARP = Registry('tps_warp')

GEOMETRICMATCHING = Registry('geometric_matching')

UNETSKIPCONNECTIONBLOCK = Registry('unet_skip_connection_block')
TRYON = Registry('tryon')

