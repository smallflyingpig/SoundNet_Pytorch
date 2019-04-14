import argparse
import numpy as np 
import torch
from collections import OrderedDict
# reference: https://github.com/smallflyingpig/SoundNet-tensorflow
# torch model: OrderDict:
# odict_keys(['BN1.weight', 'BN1.bias', 'BN1.running_mean', 'BN1.running_var', 'BN1.num_batches_tracked', 'classifier.0.weight', 'classifier.1.weight', 'classifier.1.bias', 'classifier.1
# .running_mean', 'classifier.1.running_var', 'classifier.1.num_batches_tracked', 'classifier.4.conv1.weight', 'classifier.4.bn1.weight', 'classifier.4.bn1.bias', 'classifier.4.bn1.running_mean'
# , 'classifier.4.bn1.running_var', 'classifier.4.bn1.num_batches_tracked', 'classifier.4.conv2.weight', 'classifier.4.bn2.weight', 'classifier.4.bn2.bias', 'classifier.4.bn2.running_mean', 'cla
# ssifier.4.bn2.running_var', 'classifier.4.bn2.num_batches_tracked', 'classifier.4.conv3.weight', 'classifier.4.bn3.weight', 'classifier.4.bn3.bias', 'classifier.4.bn3.running_mean', 'classifie
# r.4.bn3.running_var', 'classifier.4.bn3.num_batches_tracked', 'classifier.4.downsample.0.weight', 'classifier.4.downsample.1.weight', 'classifier.4.downsample.1.bias', 'classifier.4.downsample.1.running_mean', 'classifier.4.downsample.1.running_var', 'classifier.4.downsample.1.num_batches_tracked', 'classifier.6.conv1.weight', 'classifier.6.bn1.weight', 'classifier.6.bn1.bias', 'classifier.6.bn1.running_mean', 'classifier.6.bn1.running_var', 'classifier.6.bn1.num_batches_tracked', 'classifier.6.conv2.weight', 'classifier.6.bn2.weight', 'classifier.6.bn2.bias', 'classifier.6.bn2.running_mean', 'classifier.6.bn2.running_var', 'classifier.6.bn2.num_batches_tracked', 'classifier.6.conv3.weight', 'classifier.6.bn3.weight', 'classifier.6.bn3.bias', 'classifier.6.bn3.running_mean', 'classifier.6.bn3.running_var', 'classifier.6.bn3.num_batches_tracked', 'classifier.6.downsample.0.weight', 'classifier.6.downsample.1.weight', 'classifier.6.downsample.1.bias', 'classifier.6.downsample.1.running_mean', 'classifier.6.downsample.1.running_var', 'classifier.6.downsample.1.num_batches_tracked', 'classifier.8.weight'])

# tensor model: numpy dict:
# dict_keys(['conv3', 'conv2', 'conv1', 'conv7', 'conv6', 'conv5', 'conv4', 'conv8_2', 'conv8'])
# In [24]: data_dict['conv1'].keys()                                                                                                                                                              
# Out[24]: dict_keys(['beta', 'weights', 'biases', 'var', 'gamma', 'mean']) # (weights, bias) for conv layer, (mean, var, gamma, beta) for BN layer

# (block_name, layer_idx, (tf_param_name, pytorch_param_name))
_layer_param_dict = {
    'conv_layer':[('weights', 'weight', (3,2,0,1)), ('biases', 'bias')], #(H,W,in_channel, out_channel)-->(out_channel, in_channel, H,W)
    'batch_norm_layer':[('gamma','weight'), ('beta','bias'), ('mean','running_mean'), ('var','running_var'), (100.0, 'num_batches_tracked')]
}

g_param_dict = [
    ('conv1', '0', _layer_param_dict['conv_layer']),
    ('conv1', '1', _layer_param_dict['batch_norm_layer']),
    ('conv2', '0', _layer_param_dict['conv_layer']),
    ('conv2', '1', _layer_param_dict['batch_norm_layer']),
    ('conv3', '0', _layer_param_dict['conv_layer']),
    ('conv3', '1', _layer_param_dict['batch_norm_layer']),
    ('conv4', '0', _layer_param_dict['conv_layer']),
    ('conv4', '1', _layer_param_dict['batch_norm_layer']),
    ('conv5', '0', _layer_param_dict['conv_layer']),
    ('conv5', '1', _layer_param_dict['batch_norm_layer']),
    ('conv6', '0', _layer_param_dict['conv_layer']),
    ('conv6', '1', _layer_param_dict['batch_norm_layer']),
    ('conv7', '0', _layer_param_dict['conv_layer']),
    ('conv7', '1', _layer_param_dict['batch_norm_layer']),
    ('conv8', '0', _layer_param_dict['conv_layer']),
    ('conv8_2', '0', _layer_param_dict['conv_layer'])
]

def convert_tf2pytorch(tf_param_dict, param_dict):
    torch_param_list = []
    for param in param_dict:
        block_name, layer_idx = param[0], param[1]
        layer_param_list = param[2]
        for layer_param in layer_param_list:
            param_name_tf, param_name_torch = layer_param[0], layer_param[1]
            
            torch_param_name = '.'.join([block_name, layer_idx, param_name_torch])

            if isinstance(param_name_tf, str):
                torch_param_value = tf_param_dict[block_name][param_name_tf]
            elif isinstance(param_name_tf, (float, int)):
                torch_param_value = param_name_tf
            else:
                raise ValueError
            torch_param_value = torch.tensor(torch_param_value, device='cpu')
            if len(layer_param)>2:
                transpose_idx = layer_param[2]
                torch_param_value = torch_param_value.permute(transpose_idx)
            torch_param_list.append((torch_param_name, torch_param_value))

    return OrderedDict(torch_param_list)


def get_parser():
    parser = argparse.ArgumentParser("convert")
    parser.add_argument("--tf_param_path", type=str, default="./sound8.npy", help="")
    parser.add_argument("--pytorch_param_path", type=str, default="./sound8.pth", help="")

    args, _ = parser.parse_known_args()
    return args

def main(args):
    tf_param_path = args.tf_param_path
    pytorch_param_path = args.pytorch_param_path
    tf_param = np.load(tf_param_path, encoding='latin1').tolist()
    print("load tf param:{}".format(tf_param_path))
    pytorch_param = convert_tf2pytorch(tf_param, g_param_dict)
    torch.save(pytorch_param, pytorch_param_path)
    print("save pytorch model:{}".format(pytorch_param_path))


if __name__=="__main__":
    args = get_parser()
    main(args)