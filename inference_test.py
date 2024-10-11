import dotenv
dotenv.load_dotenv()
import numpy as np
import torch
torch.set_grad_enabled(False)
from msclap import CLAP
import onnxruntime

x_numpy = np.random.randn(1, 4*22050).astype(np.float32)
x_torch = torch.from_numpy(x_numpy)
clap = CLAP(version='2022')
clap = clap.clap
delattr(clap, 'caption_encoder') 
clap_output = clap.audio_encoder(x_torch)[0]

ort_session = onnxruntime.InferenceSession("clap.onnx", providers=["CPUExecutionProvider"])

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x_numpy}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(clap_output.detach().cpu().numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")