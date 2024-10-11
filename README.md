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

You should also create a .env file (a template can be found in template.env)

To convert CLAP into onnx, please run 
```bash
python onnx_clap.py
```

You can then check if the conversion was succesful by running 
```bash
python inference_test.py 
```