
"""
Predicts class for a target image URL from a trained model
"""

import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import argparse
import model_builder

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Creating a parser
parser = argparse.ArgumentParser()

# Get an IMAGE_URL
parser.add_argument("--image_url",
                    default="https://na-talerzu.pl/wp-content/uploads/2022/08/Paella-z-krewetkami-i-chorizo-0849-2.jpg",
                    type=str,
                    help="target image url to predict on")

# Get a model path
parser.add_argument("--model_path",
                    default="models/05_modular_approach_tinyvgg_model.pth",
                    type=str,
                    help="target model to use for prediction filepath")

# Get an HIDDEN_UNITS
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of neurons in hidden layers")

# Get the arguments from the parser
args = parser.parse_args()
IMAGE_URL = args.image_url
MODEL_PATH = args.model_path
HIDDEN_UNITS = args.hidden_units


# Setup custom image path
data_path = Path("data/")
custom_image_path = data_path / "predict_image.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
  with open(custom_image_path, "wb") as f:
    request = requests.get(IMAGE_URL)
    print(f"Downloading {custom_image_path}...")
    f.write(request.content)
else:
  print(f"{custom_image_path} already exists, overwriting content...")
  with open(custom_image_path, "wb") as f:
    request = requests.get(IMAGE_URL)
    print(f"Downloading {custom_image_path}...")
    f.write(request.content)
print("Done")


class_names = ['cheesecake', 'gnocchi', 'guacamole', 'hamburger', 'paella']

# Reload the model
try:
  model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names)).to(device)
  model.load_state_dict(torch.load(MODEL_PATH))
except:
  print("Hidden units parameter not matching the loaded model")


# Load the image
custom_image = torchvision.io.read_image(str(custom_image_path))

# Create transform pipeline to resize image
custom_image_transform = transforms.Compose([transforms.Resize(size=(224, 224), antialias=True)])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: list[str] = None,
                        transform=None,
                        device=device):
  """Makes a prediction on a target image with a trained model and plots the image and prediction."""
  # Load in the image
  target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

  # Divide the image pixel values by 255 to get them between [0, 1]
  target_image = target_image / 255.

  # Transform if necessary
  if transform:
    target_image = transform(target_image)

  # Make sure the model is on the target device
  model.to(device)

  # Turn on eval/inference mode and make a prediction
  model.eval()
  with torch.inference_mode():
    # Add an extra dimension to the image (this is the batch dimension)
    target_image = target_image.unsqueeze(0)

    # Make a prediciton on the image with an extra dimension
    target_image_pred = model(target_image.to(device))

  # Convert logits -> prediction probabilities
  target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

  # Convert prediction probabilities -> prediction labels
  target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

  # Plot the image alongside the prediction and prediction probability
  print(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max().cpu():.3f}")

# Pred on our custom image
pred_and_plot_image(model=model,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)
