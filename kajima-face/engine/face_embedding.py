# coding: utf-8
# coded by Yi Cheng

import mxnet as mx
import cv2
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import normalize
# from mxnet.contrib import onnx as onnx_mxnet


class FaceEmbedding(object):

    def __init__(self, model_path, model_epoch, device, emb_layer_name="fc1_output"):
        print("Loading Feature Extraction Model...")
        self.Batch = namedtuple('Batch', ['data'])

        # loading model
        # sym, arg_params, aux_params = mx.model.load_checkpoint(model_path, model_epoch)
        # sym, arg_params, aux_params = onnx_mxnet.import_model('/home/tanxh33/Desktop/CurrentProject/onnx/model_onnx/r34.onnx')

        jsonfile = "{}-symbol.json".format(model_path)
        sym = mx.sym.load(jsonfile)
        paramfile = "{}-{:04}.params".format(model_path, model_epoch)
        # print(paramfile)
        arg_params, aux_params = self.load_param(paramfile)


        all_layers = sym.get_internals()
        fe_sym = all_layers[emb_layer_name]  # 'fc1_output' by default

        self.fe_mod = mx.mod.Module(symbol=fe_sym, context=device, label_names=None)
        self.fe_mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
        self.fe_mod.set_params(arg_params, aux_params)

        print("Feature Extraction Loaded.")

    def load_param(self, param_path):
        node_data = mx.nd.load(param_path)
        # pickle is not working although the doc says so
        # param_file = open('./model/card-0001.params','rb')
        # print pickle.load(param_file)

        # parse node data into 2 lists
        arg_param={}
        aux_param={}
        for k, v in node_data.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_param[name] = v
            if tp == 'aux':
                aux_param[name] = v
        # print(arg_param)
        # print(aux_param)
        return arg_param, aux_param

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

