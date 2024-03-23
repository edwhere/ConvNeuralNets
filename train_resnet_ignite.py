"""Train ResNet Convolutional Neural Networks with the help of PyTorch Ignite."""

import os
import time
import torch
import argparse
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models import (ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
                                ResNet101_Weights, ResNet152_Weights)
from types import SimpleNamespace

import ignite.engine
import ignite.metrics
import ignite.handlers
import ignite.contrib.handlers

NUM_CLASSES = 2
MAX_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARN_RATE = 0.0001

TRN_DESC = ["trn", "train", "training"]
VAL_DESC = ["val", "valid", "validation"]
TRN_FOLDER = "DataTrain"
VAL_FOLDER = "DataValid"

RESNET_VARIANTS = [18, 34, 50, 101, 152]
RESNET_VARIANT = 18


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
        "e": f"Number of epochs. Default value is {NUM_EPOCHS}.",
        "m": f"ResNet model variant selected from {RESNET_VARIANTS}. Default value is {RESNET_VARIANT}"
    }

    desc = SimpleNamespace(**msg)

    required.add_argument("--root_dir", help=desc.r, type=str, required=True)
    required.add_argument("--output_dir", help=desc.o, type=str, required=True)
    parser.add_argument("--batch", help=desc.b, type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", help=desc.e, type=int, default=NUM_EPOCHS)
    parser.add_argument("--learn_rate", help=desc.l, type=float, default=LEARN_RATE)
    parser.add_argument("--variant", help=desc.m, type=int,
                        choices=RESNET_VARIANTS, default=RESNET_VARIANT)

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


class ResNetCone(nn.Module):
    def __init__(self, num_outputs, model_variant):
        super(ResNetCone, self).__init__()

        if model_variant == 18:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_variant == 34:
            self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif model_variant == 50:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif model_variant == 101:
            self.model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        elif model_variant == 152:
            self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown ResNet variant in ResNetGeneric class")

        final_layer_len = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Linear(in_features=final_layer_len, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=256, out_features=num_outputs)
        )

    def forward(self, x):
        return self.model(x)


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


def main():
    """Run main sequence of operations."""

    start_time = time.time()
    args = get_arguments()

    # Get loaders for data stored in structured directories (directories follow the PyTorch ImageFolder format)
    trn_dataset = datasets.ImageFolder(root=os.path.join(args.root_dir, TRN_FOLDER), transform=get_transform('trn'))
    val_dataset = datasets.ImageFolder(root=os.path.join(args.root_dir, VAL_FOLDER), transform=get_transform('val'))
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = len(trn_dataset.classes)

    print("Number of images in the train dataset: ", len(trn_dataset))
    print("Number of images in the valid dataset: ", len(val_dataset))
    indexing_dict = trn_dataset.class_to_idx
    print("Label indexing: ", indexing_dict)

    # Get device information
    device = get_device()
    print(f"Selected backend device: {device}")

    # Initialize a model and transfer it to the selected device
    model = ResNetCone(num_outputs=num_classes, model_variant=args.variant)
    model = model.to(device)

    # Set up an optimization criterion and the type of optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    # ----- PyTorch-Ignite manages the training and validation phases -------------

    # Define the typical evaluation metrics: accuracy and loss
    eval_metrics = {
        "acc": ignite.metrics.Accuracy(),
        "loss": ignite.metrics.Loss(criterion)
    }

    # Define trainers and evaluators
    trainer = ignite.engine.create_supervised_trainer(model, optimizer, criterion, device)
    trn_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=eval_metrics, device=device)
    val_evaluator = ignite.engine.create_supervised_evaluator(model, metrics=eval_metrics, device=device)

    # Event handler for showing training accuracy and loss after an epoch
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_training_results(sel_trainer):
        trn_evaluator.run(trn_loader)
        metrics = trn_evaluator.state.metrics
        print(f"TRN Epoch: {sel_trainer.state.epoch}, Acc: {metrics['acc']:.2f}, Loss: {metrics['loss']:.2f}")

    # Event handler for showing validation accuracy and loss after an epoch
    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def log_validation_results(sel_trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        print(f"VAL Epoch: {sel_trainer.state.epoch}, Acc: {metrics['acc']:.2f} Loss: {metrics['loss']:.2f}")

    # Function that takes an ignite engine object and defines the metric used to evaluate models
    def score_function(engine_object):
        """Define the score to evaluate models during training"""
        return engine_object.state.metrics["acc"]

    # Object that defines how checkpoints get created
    model_checkpoint = ignite.handlers.ModelCheckpoint(
        dirname=args.output_dir,
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=ignite.contrib.handlers.global_step_from_engine(trainer),
    )

    # Add another event handler for saving models
    val_evaluator.add_event_handler(ignite.engine.Events.COMPLETED, model_checkpoint, {"model": model})

    # Define a tensorboard logging object
    tb_logger = ignite.contrib.handlers.TensorboardLogger(log_dir=os.path.join(args.output_dir, "Logs"))

    # Add more event handlers, this time to add tensorboard logs after each epoch
    # There seem to be too many ways to define event handlers.
    for tag, evaluator in [("trn", trn_evaluator), ("val", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=ignite.engine.Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=ignite.contrib.handlers.global_step_from_engine(trainer),
        )

    # And finally, we can run training and validation procedures
    trainer.run(trn_loader, max_epochs=args.epochs)
    tb_logger.close()

    elapsed_time = int(time.time() - start_time)
    print("Process finished in {} seconds".format(elapsed_time))


if __name__ == "__main__":
    main()
