# import mmcv
from torch import nn

from predictors import (ATTRPREDICTOR, BACKBONES, CATEPREDICTOR, CONCATS,
                               GLOBALPOOLING, LOSSES, PREDICTOR, ROIPOOLING)


def _build_module(cfg, registry, default_args):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if isinstance(obj_type, str):
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)


def build_backbone(cfg):
    print(cfg, BACKBONES)
    return build(cfg, BACKBONES)


def build_global_pool(cfg):
    return build(cfg, GLOBALPOOLING)


def build_roi_pool(cfg):
    return build(cfg, ROIPOOLING)


def build_concat(cfg):
    return build(cfg, CONCATS)


def build_attr_predictor(cfg):
    return build(cfg, ATTRPREDICTOR)


def build_cate_predictor(cfg):
    return build(cfg, CATEPREDICTOR)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_predictor(cfg):
    return build(cfg, PREDICTOR)

