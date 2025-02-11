# oclapos

Porting Microsoft CLAP to onnx

## Installation

After activating your virtual environment, install the necessary packages by running : 
```bash
pip install -r requirements.txt
```

You should also install torch, please run 
```bash 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

You should also create a .env file (a template can be found in template.env) to specify where the weights of the models 
should be downloaded (for now, only CLAP is available)

To convert CLAP into onnx, please run 
```bash
python onnx_clap.py
```

You can then check if the conversion was succesful by running 
```bash
python inference_test.py 
```

To use this onnx-ed model to process folders of audio samples, you can run the following command : 
```bash
python extract_embeddings.py --model path/to/model.onnx --out_folder path/to/store/embeddings --sample_folder path/to/audio/samples
```
(Note that this also works with multiple folders of audio samples, just repeat the ```--sample_folder``` argument).

Finally, to create a 2-dimensional map, please run :
```bash
python make_map.py --embeddings_file path/to/embeddings.json --out_folder path/to/store/map.json
```
Note that the default map dimension is 2, but if you feel this is not enough, you can provide an extra argument ```--dim``` to specify the dimension of the map.
For instance, ```--dim 3``` will create a 3-dimensional map. 