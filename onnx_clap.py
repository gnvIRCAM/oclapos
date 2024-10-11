import dotenv
dotenv.load_dotenv()
import torch
import torch.onnx
from msclap import CLAP

# Load model (Choose between versions '2022' or '2023')
# The model weight will be downloaded automatically if `model_fp` is not specified
clap = CLAP(version='2022')
clap = clap.clap
delattr(clap, 'caption_encoder') 
# delattr(clap, 'audio_encoder.base.spectrogram_extractor') 
# delattr(clap, 'audio_encoder.base.logmel_extractor') 

x = torch.randn(1, 22050)
y, _ = clap.audio_encoder(x)
clap.forward = clap.audio_encoder.forward

torch.onnx.export(
    clap, 
    x, 
    'clap_stupiderer.onnx', 
    export_params=True,        
    opset_version=17,          
    do_constant_folding=True, 
    input_names = ['input'],   
    output_names = ['output'], 
    dynamic_axes={'input' : {0 : 'batch_size'},    
                'output' : {0 : 'batch_size'}}
    )