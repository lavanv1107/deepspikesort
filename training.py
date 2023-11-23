import os

import torch
import torch.nn as nn

import time
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from util import AverageMeter 


class TrainModel():
    """
    A PyTorch-based class for training and validating a neural network model. 
    It utilizes DataParallel for training on multiple GPUs.
    """

    def __init__(self, device, device_ids, loss_fn, optimizer, train_dataloader=None, test_dataloader=None):
        """
        Initializes the training and testing setup.

        Args:
            device (str): The device on which to run the model (e.g., 'cuda:0').
            device_ids (list): List of GPU device IDs to use for DataParallel.
            loss_fn (torch.nn.modules.loss): The loss function to use for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            train_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the training data.
            test_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the testing data.
        """
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.device_ids = device_ids
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, model, epoch, verbose_interval=10):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to train.
            epoch (int): The current epoch number.
            verbose_interval (int, optional): Interval (in number of batches) at which to print detailed training progress. 
                                               If set to 0 or None, no detailed progress will be printed. Defaults to 5.

        Returns:
            float: The average loss for an epoch.
        """    
        loss_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()
        
        # Start training the model
        model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        model.train()

        # Iterate through each batch in the training dataloader
        end = time.time()
        for batch, (X, Y) in enumerate(self.train_dataloader):
            data_time_meter.update(time.time() - end)

            # Move input data and labels to the specified device (e.g., GPU)
            X, Y = X.to(self.device), Y.to(self.device)

            # Zero out the gradients in the model's parameters
            self.optimizer.zero_grad()

            # Forward pass: Compute predictions using the model
            inputs_dict = {'x': X}
            pred = model(inputs_dict)

            # Compute the loss between the predictions and the ground truth labels
            loss = self.loss_fn(pred, Y)

            # Backpropagation: Compute gradients of the loss with respect to model parameters
            loss.backward()

            # Update the model's parameters using the optimizer
            self.optimizer.step()
            
            loss_meter.update(loss.item(), X.size(0))
            batch_time_meter.update(time.time() - end)
            end = time.time()
            
            if verbose_interval and (batch % verbose_interval) == 0:
                print('- Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch+1, batch, len(self.train_dataloader), batch_time=batch_time_meter,
                              data_time=data_time_meter, loss=loss_meter))
            
        return loss_meter.avg
        
    def test(self, model):
        """
        This tests/validates the model and prints the model's accuracy and average loss for an epoch.
        It also returns the true labels and predicted labels for an epoch as well.

        Args:
            model (obj): A PyTorch model.
            
        Returns:
            test_loss (int): The average test loss for an epoch.
            true_labels (obj): A list of the actual true labels being used for training.
            predicted_labels (obj): A list of the model's predicted labels when validating.
        """
        # Get the total number of samples in the test dataset
        size = len(self.test_dataloader.dataset)
        
        # Set the model to evaluation mode (no gradient updates during evaluation)        
        model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        model.eval()

        # Initialize variables to keep track of test loss and correct predictions
        test_loss, correct = 0, 0

        # Lists to store true labels and predicted labels for later analysis
        true_labels = []
        predicted_labels = []

        # Disable gradient computation for efficiency during evaluation
        with torch.no_grad(), tqdm(total=len(self.test_dataloader), desc='Testing', dynamic_ncols=True) as progress_bar:
            # Iterate through each batch in the test dataloader
            for batch, (X, Y) in enumerate(self.test_dataloader):

                # Move input data and labels to the specified device (e.g., GPU)
                X, Y = X.to(self.device), Y.to(self.device)

                # Forward pass: Compute predictions using the model
                inputs_dict = {'x': X}
                pred = model(inputs_dict)

                # Compute the test loss and add it to the running total
                test_loss += self.loss_fn(pred, Y).item()

                # Count correct predictions by comparing predicted labels to true labels
                correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

                # Store the true labels and predicted labels for further analysis
                true_labels.extend(Y.tolist())
                predicted_labels.extend(pred.argmax(1).tolist())

                # Update the progress bar
                progress_bar.update()

        # Average test loss over the number of batches
        test_loss /= size
        correct /= size

        # Print the test results (accuracy and average loss)
        print(f'\nTest Error:\n- Accuracy: {(100 * correct):>0.1f}%\n- Average Loss: {test_loss:>8f}\n')

        # Return the test loss, true labels, and predicted labels for external analysis
        return test_loss, true_labels, predicted_labels


    def train_test_model(self, model, model_name, models_folder, epochs, classes):
        """
        This trains and tests a model.
        It plots and saves loss and accuracy over time as well as a confusion matrix for the model's predictions.
        There is also a checkpoint saving functionality which allows for pausing the training which saves the model's 
        current state, current epoch and accumulated losses and accuriacies.

        Args:
            model (obj): A PyTorch model
            model_name (str): A given name for the model.
            models_folder (str): The folder path name containing the model.
            epochs (int): The number of epochs to train the model.
            classes (int): The classes in the dataset.
        """
        model, start_epoch, losses, accuracies = load_checkpoint(model, model_name, models_folder)
        
        for epoch in range(start_epoch, epochs):
            # print(f'Epoch {epoch+1}\n-------------------------------')

            # Train the model and get the losses for this epoch
            self.train(model)

            # Test the model and get the loss and accuracy for this epoch
            test_loss, true_labels, predicted_labels = self.test(model)
            
            cm = confusion_matrix(true_labels, predicted_labels, normalize='all')
            plt.figure(figsize=(30, 30))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix ({epoch+1} Epochs)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(os.path.join(models_folder, f'{model_name}_confusion_matrix.png'))
        
        print('Training completed.\n')


def save_checkpoint(model, model_name, models_folder, end_epoch, losses, accuracies):
    """
    Creates a dictionary of checkpoint variables for a model.
    This inlcudes the model's state dictionary, the epoch in which it finished/stopped training, and the accumulated losses and accuracies from training.
    The checkpoint will be saved to disk as a PyTorch pt file.

    Args:
        model (obj): A PyTorch model.
        model_name (str): A given name for the model.
        models_folder (str): The folder path name containing the model.
        end_epoch (int): The epoch when training was finished/stopped.
        losses (obj): A list of losses per epoch from training.
        accuracies (obj): A list of accuracies per epoch from training.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'end_epoch': end_epoch,
        'losses': losses,
        'accuracies': accuracies
    }
    checkpoint_file = os.path.join(models_folder, f'{model_name}.pt')
    torch.save(checkpoint, checkpoint_file)
    print(f'Checkpoint saved: {checkpoint_file}\n')


def load_checkpoint(model, model_name, models_folder):
    """
    Loads a checkpoint for a model from a previous training.
    This will return the model with its previous state dictionary, the epoch to continue training from and the accumulated losses and accuracies from previous training.

    Args:
        model (obj): Number of images used for spikes per spike unit.
        model_name (int): Number of images used for noise.
        models_folder (str): The folder path name of the model.
        
    Returns:
        model (obj): A PyTorch model.
        start_epoch (int): The epoch to continue training from. 
        losses (obj): A list of losses per epoch from training.
        accuracies (obj): A list of accuracies per epoch from training.
    """
    checkpoint_file = os.path.join(models_folder, f'{model_name}.pt')
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['end_epoch']
        losses = checkpoint['losses']
        accuracies = checkpoint['accuracies']
        print(f'Checkpoint loaded: {checkpoint_file}\nEnded training at epoch {start_epoch}\n')
        return model, start_epoch, losses, accuracies
    else:
        print('No checkpoint found.\nStart training at epoch 1\n')
        return model, 0, [], []
