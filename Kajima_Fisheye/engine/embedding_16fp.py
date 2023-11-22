# coding: utf-8
# coded by Yi Cheng

import mxnet as mx
from mxnet.contrib import amp
import cv2
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import normalize

class FaceEmbedding(object):

    def __init__(self, model_path, model_epoch, device, input_size, emb_layer_name="fc1_output"):
        print("Loading Feature Extraction Model...")

        self.mx_device = mx.cpu(0)
        if device >= 0:
            self.mx_device = mx.gpu(device)

        self.Batch = namedtuple('Batch', ['data'])
        # loading model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym, arg_params, aux_params)

        # sym, arg_params, aux_params = onnx_mxnet.import_model('/home/tanxh33/Desktop/CurrentProject/onnx/model_onnx/r34.onnx')

        all_layers = result_sym.get_internals()
        fe_sym = all_layers[emb_layer_name]  # 'fc1_output' by default

        self.fe_mod = mx.mod.Module(symbol=fe_sym, context=self.mx_device, label_names=None)
        self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, input_size, input_size))])
        self.fe_mod.set_params(result_arg_params, result_aux_params)

        print("Feature Extraction Loaded.")

    def extract_feature(self, input_image, normalise_features=False):
        #img = cv2.resize(input_image, (112, 112))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        self.fe_mod.forward(self.Batch([mx.nd.array(img)]))
        feature = self.fe_mod.get_outputs()[0].asnumpy()

        # print("out shape", feature.shape)
        # print(feature[:, :6])

        if not normalise_features:
            return feature

        feature_norm = normalize(feature)
        # print(feature_norm[:, 0:10])
        # print(feature_norm.shape)
        return feature_norm

    def extract_feature_batch(self, img_list):
        images = np.swapaxes(img_list, 1, 3)
        images = np.swapaxes(images, 2, 3)

        self.fe_mod.forward(self.Batch([mx.nd.array(images)]))
        features = self.fe_mod.get_outputs()[0].asnumpy()
        # print("out shape:", features.shape)
        # print(features[:, :6])

        return features

class UbodyEmbedding(object):

    def __init__(self, model_path, model_epoch, device, input_size, emb_layer_name="fc1_output"):
        print("Loading Feature Extraction Model...")

        self.mx_device = mx.cpu(0)
        if device >= 0:
            self.mx_device = mx.gpu(device)

        self.Batch = namedtuple('Batch', ['data'])
        # loading model
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        result_sym, result_arg_params, result_aux_params = amp.convert_model(sym, arg_params, aux_params)
        # sym, arg_params, aux_params = onnx_mxnet.import_model('/home/tanxh33/Desktop/CurrentProject/onnx/model_onnx/r34.onnx')
        all_layers = result_sym.get_internals()
        fe_sym = all_layers[emb_layer_name]  # 'fc1_output' by default

        self.fe_mod = mx.mod.Module(symbol=fe_sym, context=self.mx_device, label_names=None)
        # self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
        self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, input_size, input_size))])
        self.fe_mod.set_params(result_arg_params, result_aux_params)

        print("Feature Extraction Loaded.")

    def extract_feature(self, input_image, normalise_features=False):
        #img = cv2.resize(input_image, (112, 112))
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img = img[np.newaxis, :]

        self.fe_mod.forward(self.Batch([mx.nd.array(img)]))
        feature = self.fe_mod.get_outputs()[0].asnumpy()

        # print("out shape", feature.shape)
        # print(feature[:, :6])

        if not normalise_features:
            return feature

        feature_norm = normalize(feature)
        # print(feature_norm[:, 0:10])
        # print(feature_norm.shape)
        return feature_norm

    def extract_feature_batch(self, img_list):
        images = np.swapaxes(img_list, 1, 3)
        images = np.swapaxes(images, 2, 3)

        self.fe_mod.forward(self.Batch([mx.nd.array(images)]))
        try:
            features = self.fe_mod.get_outputs()[0].asnumpy()
        except ValueError:
            features = None
        # print("out shape:", features.shape)
        # print(features[:, :6])

        return features

