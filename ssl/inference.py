import torch
import gradio as gr
import os
import torchvision.transforms as T
import numpy as np

from PIL import Image
from conv_auto_encoder import ConvAutoEncoder

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

model = ConvAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load('./model/conv_auto_encoder.pth'))
model.eval()

transform = T.ToTensor()

def image_mod(image):
    output = model(transform(image.convert('RGB')).unsqueeze(0).to(DEVICE))
    output = output.squeeze(0)
    numpy_img = np.transpose(output.cpu().detach().numpy(), (1, 2, 0))
    return Image.fromarray((numpy_img * 255).astype(np.uint8))

demo = gr.Interface(
    image_mod,
    gr.Image(type="pil"),
    "image",
    flagging_options=["blurry", "incorrect", "other"],
)

if __name__ == "__main__":
    demo.launch()