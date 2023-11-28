"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils
from timeit import default_timer as timer

# Create a parser
parser = argparse.ArgumentParser(description="Get hyperparameters")

# Get an arg for NUM_EPOCHS
parser.add_argument("--num_epochs",
                    default=5,
                    type=int,
                    help="number of epochs to train for")

# Get an arg for BATCH_SIZE
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="number of samples per batch")

# Get an arg for HIDDEN_UNITS
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help="number of neurons in hidden layers")

# Get an arg for LEARNING_RATE
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help="learning rate for the optimizer")

# Get an arg for train_dir
parser.add_argument("--train_dir",
                    default="data/food101_extract/train",
                    type=str,
                    help="directory file path to training data in standard image classification format")

# Get an arg for test_dir
parser.add_argument("--test_dir",
                    default="data/food101_extract/test",
                    type=str,
                    help="directory file path to testing data in standard image classification format")

# Get the arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")


# Setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f"[INFO] Training data file: {train_dir}")
print(f"[INFO] Testing data file: {test_dir}")

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
])


# Create DataLoaders and get class_names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                               test_dir=test_dir,
                                                                               transform=data_transform,
                                                                               batch_size=BATCH_SIZE)

# Create a model
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=HIDDEN_UNITS,
                              output_shape=len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the timer
start_time = timer()


# Start training with help from engine.py
model_results = engine.train(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             optimizer=optimizer,
                             loss_fn=loss_fn,
                             epochs=NUM_EPOCHS,
                             device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


# Save the model to file
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_modular_approach_tinyvgg_model.pth")
print("model saved")

