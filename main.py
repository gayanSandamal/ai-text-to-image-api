from typing import Union

from fastapi import FastAPI

app = FastAPI()

from diffusers import DiffusionPipeline
import torch

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
base.to("cuda")
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

def generate_image(prompt):
  # run both experts
  image = base(
      prompt=prompt,
      num_inference_steps=n_steps,
      denoising_end=high_noise_frac,
      output_type="latent",
  ).images
  image = refiner(
      prompt=prompt,
      num_inference_steps=n_steps,
      denoising_start=high_noise_frac,
      image=image,
  ).images[0]

  image.show()
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
    return generate_images(prompt)