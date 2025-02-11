from absl import flags, app
import os 

import librosa as li
import numpy as np
import onnxruntime

import soundfile as sf
import torch
torch.set_grad_enabled(False)
from tqdm import tqdm

import json

_AVAILABLE_EXTS =  [k.lower() for k in  sf.available_formats().keys()]

FLAGS = flags.FLAGS
flags.DEFINE_multi_string('sample_folder', 
                          default=None, 
                          required=True, 
                          help='Folder(s) containing audio samples')
flags.DEFINE_string('out_folder', 
                    default=None, 
                    required=True, 
                    help='Path to save embeddings')
flags.DEFINE_string('model', 
                    default=None, 
                    required=True, 
                    help='Path to models onnx')

def has_valid_ext(filename):
    return np.any([filename.endswith(ext) for ext in _AVAILABLE_EXTS])

def main(argv):
    # First we load the onnx model
    ort_session = onnxruntime.InferenceSession(FLAGS.model, providers=["CPUExecutionProvider"])

    audio_paths = []
    samples_folders = FLAGS.sample_folder
    
    for folder in samples_folders:
        for root, _, _files in os.walk(folder):
            for f in tqdm(_files, leave=False, desc='Gathering samples'):
                if has_valid_ext(f):
                    audio_paths.append(os.path.join(root, f)) 
    
    audio_paths = list(set(audio_paths))
    embeddings = {}
    
    for file in tqdm(audio_paths, desc='Processing samples'):
        try:
            y, sr = li.load(path=file)
        except:
            # Dirty skipping audio files that could not be loaded 
            continue
        
        # Stereo -> Mono
        if y.shape[0]==2:
            y = y.mean(0)[None, :]
            
        # Resample to CLAP sample rate
        if sr!=22050:
            y = li.resample(y, sr, 22050)
        
        # Need "batch" axis
        if y.ndim==1:
            y = y[None, :]
        
        # Minimal duration for CLAP seems to be 0.5 seconds, so we need to pad everything
        if y.shape[-1]<int(.5*22050): 
            delta = int(.5*22050)-y.shape[-1]
            y = np.concatenate((y, np.zeros((1, delta))), axis=-1)
        
        y = y.astype(np.float32)
        
        # Extract embeddings
        ort_inputs = {ort_session.get_inputs()[0].name: y}
        ort_outs = ort_session.run(None, ort_inputs)
        embeddings[file] = ort_outs[0].tolist()
    
    os.makedirs(FLAGS.out_folder, exist_ok=True)
    with open(os.path.join(FLAGS.out_folder, 'embeddings.json'), 'w') as f:
        json.dump(embeddings, f)
        

if __name__=='__main__':
    app.run(main)