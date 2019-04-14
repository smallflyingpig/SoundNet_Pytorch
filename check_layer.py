import argparse
import numpy as np
import os
import tensorflow as tf

tf_feature_layers = [4,8,11,14,18,21,24,25,26]

local_config = {  
            'batch_size': 1, 
            'eps': 1e-5,
            'sample_rate': 22050,
            'load_size': 22050*20,
            'name_scope': 'SoundNet',
            'phase': 'extract',
            }

 #   # Init. Session
 #   sess_config = tf.ConfigProto()
 #   sess_config.allow_soft_placement=True
 #   # sess_config.gpu_options.allow_growth = True
 #   
 #   
 #   with tf.Session(config=sess_config) as session:
 #       # Build model
 #       model = Model(session, config=local_config, param_G=param_G)
 #       init = tf.global_variables_initializer()
 #       session.run(init)
 #       
 #       model.load()
 #   
 #       for idx, sound_sample in enumerate(sound_samples):
 #           output = extract_feat(model, sound_sample, args)

def extract_tf_feature(input_data:np.ndarray, tf_param_path:str)->list:
    from tf_model import SoundNet8_tf

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    tf_param = np.load(tf_param_path, encoding='latin1').item()
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    with tf.Session(config=sess_config) as session:
        # Build model
        model = SoundNet8_tf(session, config=local_config, param_G=tf_param)
        init = tf.global_variables_initializer()
        session.run(init)
        model.load()
        # Demo
        sound_input = np.reshape(input_data, [1, -1, 1, 1])
        feed_dict = {model.sound_input_placeholder: sound_input}
        
        feature_all = []
        # Forward
        for idx in tf_feature_layers:
            feature = session.run(model.layers[idx], feed_dict=feed_dict)
            feature_all.append(feature)
    return feature_all
            

def extract_pytorch_feature(input_data:np.ndarray, pytorch_param_path:str)->list:
    import torch
    from pytorch_model import SoundNet8_pytorch
    # "point invalid" error, if put the import on the top of this file
    model = SoundNet8_pytorch()
    
    model.load_state_dict(torch.load(pytorch_param_path))
    
    data = torch.from_numpy(input_data).view(1,1,-1,1)
    model.eval()
    with torch.no_grad():
        feature_all = model.extract_feat(data)
    return feature_all


def get_parser():
    parser = argparse.ArgumentParser("check")
    parser.add_argument("--input_demo_data", type=str, default="./demo.npy")
    parser.add_argument("--tf_param_path", type=str, default="./sound8.npy")
    parser.add_argument("--pytorch_param_path", type=str, default="./sound8.pth")

    args, _ = parser.parse_known_args()
    return args


def main(args):
    input_data = np.load(args.input_demo_data)
    print("extract features using tensorflow model...")
    feature_all_tf = extract_tf_feature(input_data, args.tf_param_path)
    print("extrach features using pytorch model...")
    feature_all_pytorch = extract_pytorch_feature(input_data, args.pytorch_param_path)
    # check param
    layer_error_all = []
    for idx, (feat_tf, feat_pytorch) in enumerate(zip(feature_all_tf, feature_all_pytorch)):
        layer_error = feat_tf.mean()-feat_pytorch.mean()
        layer_error_all.append(layer_error)
    print("layer error:")
    print(layer_error_all)

# test
if __name__=="__main__":
    args = get_parser()
    main(args)