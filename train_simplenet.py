"""Train a simple convolutional neural network."""

import os
import json
import platform
import pandas as pd
import torch
import argparse
from types import SimpleNamespace
from torch import optim
import torch.nn as nn
import torch.nn.functional as nnfun
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from time import time

NUM_CLASSES = 2
MAX_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARN_RATE = 0.0001

TRN_DESC = ["trn", "train", "training"]
VAL_DESC = ["val", "valid", "validation"]
TRN_FOLDER = "DataTrain"
VAL_FOLDER = "DataValid"


class DirectoryAlreadyExistsError(Exception):
    """Raise an exception if a directory already exists. """
    pass


class DirectoryNotFoundError(Exception):
    """Raise an exception if a directory does not exist."""
    pass


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Train a simple CNN")
    required = parser.add_argument_group("required arguments")

    msg = {
        "r": f"Path to root directory containing {TRN_FOLDER} and {VAL_FOLDER} folders.",
        "o": "Path to output directory with model information for inference.",
        "b": f"Batch size (number of images in a batch). Default value is {BATCH_SIZE}.",
        "l": f"Learning rate. A real number in (0, 1). Default value is {LEARN_RATE}.",
        "e": f"Number of epochs. Default value is {NUM_EPOCHS}."
    }

    desc = SimpleNamespace(**msg)

    required.add_argument("--root_dir", help=desc.r, type=str, required=True)
    required.add_argument("--output_dir", help=desc.o, type=str, required=True)
    parser.add_argument("--batch", help=desc.b, type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", help=desc.e, type=int, default=NUM_EPOCHS)
    parser.add_argument("--learn_rate", help=desc.l, type=float, default=LEARN_RATE)

    args = parser.parse_args()
    verify_arguments(args)
    return args


def verify_arguments(args):
    """Check if there are obvious errors in arguments."""
    if not os.path.isdir(args.root_dir):
        raise DirectoryNotFoundError(f"Unavailable root directory: {args.root_dir}")

    if os.path.isdir(args.output_dir):
        raise DirectoryAlreadyExistsError(f"Results directory already exists: {args.output_dir}")

    if args.batch < 1:
        raise ValueError("Batch size must be a positive integer.")

    if args.epochs < 1:
        raise ValueError("Number of epochs must be a positive integer")

    if args.learn_rate <= 0 or args.learn_rate >= 1:
        raise ValueError("Learning rate must be in the interval (0, 1). Notice that bounds are excluded.")


def get_transform(subset: str):
    """Return a function that transforms images into tensors applying necessary modifications.
    ARGS:
        subset: Can be any of 'trn', 'train', 'training', 'val', 'valid', or 'validation'
    Returns:
        A transform function for training or validation
    """

    trn_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
    ])

    if subset in TRN_DESC:
        return trn_transform
    elif subset in VAL_DESC:
        return val_transform
    else:
        raise ValueError("Unknown subset string passed to get_transform_function()")


def get_device():
    """Determine if a machine runs on a gpu or cpu. Includes M1/M2/M3 gpus. Return the device
    identifier for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)

        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.25)

        self.__num_classes = num_classes

    @property
    def identifier(self):
        return "simplecnn"

    @property
    def num_classes(self):
        return self.__num_classes

    def forward(self, x):
        x = self.pool(nnfun.relu(self.conv1(x)))
        x = self.pool(nnfun.relu(self.conv2(x)))
        x = self.pool(nnfun.relu(self.conv3(x)))
        x = self.pool(nnfun.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = nnfun.adaptive_avg_pool2d(x, output_size=1).reshape(bs, -1)
        x = nnfun.relu(self.fc1(x))
        x = self.dropout(x)
        x = nnfun.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()

    train_loss = 0
    train_total = 0
    train_correct = 0

    # We loop over the data iterator, and feed the inputs to the network and adjust the weights.
    for batch_idx, (data, target) in enumerate(train_loader, start=0):
        # Load the input features and labels from the training dataset
        data, target = data.to(device), target.to(device)

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Forward pass: Pass image data from training dataset and make predictions
        output = model(data)

        # Define o loss function, and compute the loss
        loss = criterion(output, target)
        train_loss += loss.item()

        scores, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))

        # Reset the gradients to 0 for all learnable weight parameters
        optimizer.zero_grad()

        # Backward pass: compute the gradients of the loss w.r.t. the model"s parameters
        loss.backward()

        # Update the neural network weights
        optimizer.step()

    acc = round((train_correct / train_total) * 100, 2)
    print("Epoch [{}], trn_loss: {}, trn_acc: {}".format(epoch, train_loss / train_total, acc), end="")
    return train_loss / train_total, acc


def valid(model, device, test_loader, criterion):
    model.eval()

    test_loss = 0
    test_total = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            # Load the input features and labels from the test dataset
            data, target = data.to(device), target.to(device)

            # Make predictions: Pass image data from test dataset and make predictions
            output = model(data)

            # Compute the loss sum up batch loss
            test_loss += criterion(output, target).item()

            scores, predictions = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += int(sum(predictions == target))

    acc = round((test_correct / test_total) * 100, 2)
    print(" val_loss: {}, val_acc: {}".format(test_loss / test_total, acc))
    return test_loss / test_total, acc


def run_procedure():
    """Run main sequence."""
    start_time = time()
    args = get_arguments()

    # Get loaders for data stored in structured directories (follow the PyTorch ImageFolder format)
    trn_dataset = datasets.ImageFolder(root=os.path.join(args.root_dir, TRN_FOLDER), transform=get_transform('trn'))
    val_dataset = datasets.ImageFolder(root=os.path.join(args.root_dir, VAL_FOLDER), transform=get_transform('val'))
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(trn_dataset.classes)

    print("Number of images in the train dataset: ", len(trn_dataset))
    print("Number of images in the valid dataset: ", len(val_dataset))
    print("Label indexing: ", trn_dataset.class_to_idx)

    # Get device information
    device = get_device()
    print(f"Selected backend device: {device}")

    # Initialize a model and transfer it to the selected device
    model = SimpleCNN(num_classes)
    model = model.to(device)

    # Set up an optimization criterion and the type of optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

    # Run training algorithm for a number of epochs
    res_df = pd.DataFrame(columns=['epoch', 'trn_loss', 'trn_acc', 'val_loss', 'val_acc'])
    for epoch in range(args.epochs):
        trn_loss, trn_acc = train(model, device, trn_loader, optimizer, criterion, epoch)
        val_loss, val_acc = valid(model, device, val_loader, criterion)
        res_df.loc[epoch] = [epoch, trn_loss, trn_acc, val_loss, val_acc]

    # Save training context and training results
    elapsed_time = int(time() - start_time)
    current = datetime.now(timezone.utc)
    context = {
        "utc": current.strftime("%Y-%m-%dT%H:%M:%S+000Z"),
        "elapsed": elapsed_time,
        "dataset": args.root_dir,
        "host": platform.uname()[1],
        "device": device,
        "model_type": model.identifier,
        "num_classes": model.num_classes,
        "epochs": args.epochs,
        "learn_rate": args.learn_rate,
        "batch_size": args.batch,
        "optimizer": "Adam"
    }

    save_results(output_dir=args.output_dir, model=model, context=context, history=res_df)
    print("Finished training SimpleNet")


def save_results(output_dir, model, context, history):
    """Save results of training procedures.
    Args:
        output_dir: Path to output directory that contains the results
        model: Trained model
        context: Dictionary with contextual information
        history: Pandas DF with per-epoch loss and accuracy
    """
    os.makedirs(output_dir)

    # Save contextual information
    context_file = os.path.join(output_dir, "context.json")
    with open(context_file, "w") as fh:
        json.dump(context, fh, indent=4)

    # Save learning history
    history_file = os.path.join(output_dir, "progression.txt")
    history.to_csv(history_file, encoding="utf-8", index=False)

    # Save trained model
    model_file = os.path.join(output_dir, "trained_model.pth")
    torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    run_procedure()

