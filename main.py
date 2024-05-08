from typing import Union
from fastapi import FastAPI
import torch
from diffusers import StableDiffusionPipeline

app = FastAPI()

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

def generate_image(prompt):
    image = pipe(prompt).images[0]
    display(image)

def generate_images(prompt, copies_of_images = 1):
  i = 0
  while i < copies_of_images:
    generate_image(prompt)
    i += 1

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/generate/{prompt}")
def read_item(prompt: str, q: Union[str, None] = None):
    return generate_images(prompt, 3)