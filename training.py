import os

import time

import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import util 


class TrainModel():
    """
    A PyTorch-based class for training and validating a neural network model. 
    It utilizes DataParallel for training on multiple GPUs.
    """

    def __init__(self, device, device_ids, loss_fn, optimizer, train_dataloader=None, val_dataloader=None, verbose_count=25):
        """
        Initializes the training and validation setup.

        Args:
            device (str): The device on which to run the model (e.g., 'cuda:0').
            device_ids (list): List of GPU device IDs to use for DataParallel.
            loss_fn (torch.nn.modules.loss): The loss function to use for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            train_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the training data.
            val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for the validation data.
        """
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.device_ids = device_ids
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.verbose_count = verbose_count

    def train(self, model, epoch):
        """
        Args:
            model (torch.nn.Module): PyTorch model to train.
            epoch (int): Current epoch number.
            verbose_count (int, optional): Number of times to print detailed progress.

        Returns:
            float: The average loss for an epoch.
        """    
        loss_meter = util.AverageMeter()
        batch_time_meter = util.AverageMeter()
        data_time_meter = util.AverageMeter()
        
        # Start training the model
        model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        model.train()

        # Iterate through each batch in the training dataloader
        total_batches = len(self.train_dataloader)
        verbose_interval = max(1, total_batches // self.verbose_count)
        
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
            
            if batch % verbose_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, batch, total_batches, batch_time=batch_time_meter,
                              data_time=data_time_meter, loss=loss_meter))
            
        return loss_meter.avg
        
    def validate(self, model, epoch):
        """
        This validates the model and prints the model's accuracy and average loss for an epoch.
        It also returns the true labels and predicted labels for an epoch as well.

        Args:
            model (torch.nn.Module): A PyTorch model.
            
        Returns:
            val_loss (float): The average validation loss for an epoch.
            true_labels (list): The actual true labels.
            predicted_labels (list): The model's predicted labels.
        """  
        loss_meter = util.AverageMeter()
        batch_time_meter = util.AverageMeter()
        data_time_meter = util.AverageMeter()
        
        # Set the model to evaluation mode
        model.to(self.device)
        model = nn.DataParallel(model, device_ids=self.device_ids)
        model.eval()

        # Lists to store true and predicted labels
        true_labels = []
        predicted_labels = []
        
        # Total number of samples and batches for verbosity control
        total_batches = len(self.val_dataloader)
        verbose_interval = max(1, total_batches // self.verbose_count)  # Control print frequency

        # Start timing
        end = time.time()
        for batch, (X, Y) in enumerate(self.val_dataloader):
            # Update data loading time
            data_time_meter.update(time.time() - end)

            # Move data to the specified device
            X, Y = X.to(self.device), Y.to(self.device)

            # Forward pass: compute predictions
            inputs_dict = {'x': X}
            pred = model(inputs_dict)

            # Compute loss
            loss = self.loss_fn(pred, Y)
            loss_meter.update(loss.item(), X.size(0))

            # Update batch processing time
            batch_time_meter.update(time.time() - end)
            end = time.time()

            # Store labels for analysis
            true_labels.extend(Y.tolist())
            predicted_labels.extend(pred.argmax(1).tolist())

            # Print progress
            if batch % verbose_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, batch, total_batches, batch_time=batch_time_meter,
                              data_time=data_time_meter, loss=loss_meter))
                
        # Calculate accuracy
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels) if true_labels else 0

        return loss_meter.avg, accuracy, true_labels, predicted_labels


    def train_validate(self, model, num_epochs, labels_map, session_name, results_folder):
        """
        This trains and validates a model.
        It plots and saves loss and accuracy over time as well as a confusion matrix for the model's predictions.
        There is also a checkpoint saving functionality which allows for pausing the training which saves the model's 
        current state, current epoch and accumulated losses and accuriacies.

        Args:
            model (obj): A PyTorch model
            session_name (str): A given name for the model.
            results_folder (str): The folder path name containing the model.
            epochs (int): The number of epochs to train the model.
            classes (int): The classes in the dataset.
        """
        metrics_progress_file = os.path.join(results_folder, f'{session_name}_metrics_progress.log')        
        logger_class = logging.getLogger('logger_class')
        logger_class.setLevel(logging.INFO)
        file_handler_class = logging.FileHandler(metrics_progress_file, mode='w')
        formatter_class = logging.Formatter('%(asctime)s - %(message)s')
        file_handler_class.setFormatter(formatter_class)
        logger_class.addHandler(file_handler_class)
        
        losses = []
        accuracies = []
        
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        patience_limit = 10
        improvement_threshold = 0.01
    
        total_end = time.time()  
        for epoch in range(1, num_epochs+1):           
                print(util.write_separator())
                print(f"Epoch {epoch}")  
                print(util.write_separator())

                # Train the model and get the losses for this epoch
                print("- Training")
                print('Training model...')
                
                train_end = time.time()
                
                self.train(model, epoch)
                
                print(f'\nTraining time: {time.time() - train_end:.3f}\n')

                # Validate the model and get the loss and accuracy for this epoch
                print("- Validation")
                print('Validating model...')
                
                val_end = time.time()
                
                val_loss, val_acc, true_labels, predicted_labels = self.validate(model, epoch)
                losses.append(val_loss)
                accuracies.append(val_acc)
                
                log_metric_progress(logger_class, val_loss, val_acc, epoch, results_folder)  
                
                print(f'\nValidation time: {time.time() - val_end:.3f}')
                
                # Check for improvement
                if val_loss < best_loss * (1 - improvement_threshold):
                    best_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping condition
                if patience_counter >= patience_limit:
                    break
                
        print(f"\nTotal time: {time.time() - total_end:.3f}")
        
        # Visualize the changes in cluster assignments   
        metrics = {
            'Accuracy': accuracies,           
        }
        plot_metric_progress(metrics, epoch, session_name, results_folder)
        plot_confusion_matrix(true_labels, predicted_labels, labels_map, session_name, results_folder)


def log_metric_progress(logger_dss, loss, accuracy, epoch, results_folder):       
    logger_dss.info('Epoch: [{0}]\t'
                    'Loss: {1:.4f}\t'
                    'Accuracy: {2:.4f}'
                    .format(epoch, loss, accuracy))
    

def plot_metric_progress(metrics, num_epochs, session_name, results_folder):
    # Calculate the width of the figure based on the number of epochs
    base_width = 10
    dynamic_width = (num_epochs / 100) * base_width

    # Ensure a minimum width is maintained
    dynamic_width = max(base_width, dynamic_width)

    # Number of subplots
    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(dynamic_width, 5 * num_metrics))

    if num_metrics == 1:
        axs = [axs]

    for ax, (metric_type, metric_info) in zip(axs, metrics.items()):
        # Plot each metric
        ax.plot(range(1, num_epochs + 1), metric_info, marker='o', color='r')
        ax.set_title(f'{metric_type}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_type)
        ax.grid(True)

        ax.set_xlim(1, num_epochs)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"{session_name}_metrics_progress.png"))
    plt.close(fig)
    
        
def plot_confusion_matrix(true_labels, predicted_labels, labels_map, session_name, results_folder):
    """
    Plots a confusion matrix based on true and predicted labels.

    Args:
        true_labels (list): List of true numerical labels.
        predicted_labels (list): List of predicted numerical labels.
        labels_map (dict): Dictionary mapping numerical labels to class names.
        results_folder (str): Folder to save the confusion matrix image.
        session_name (str): Name of the session (used for saving the image).
    """
    # Convert numerical labels to class names using the labels map
    true_labels = [labels_map[label] for label in true_labels]
    predicted_labels = [labels_map[label] for label in predicted_labels]
    classes = sorted(labels_map.values())

    # Generate the confusion matrix with normalization
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes, normalize='true')
    formatted_cm = np.array([[format_value(value) for value in row] for row in cm])

    # Dynamic figure size
    num_classes = len(classes)
    dynamic_size = max(7.5, num_classes * 0.75)

    plt.figure(figsize=(dynamic_size, dynamic_size))
    sns.heatmap(cm, annot=formatted_cm, fmt="s", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(results_folder, f'{session_name}_confusion_matrix.png'))
    
    
def format_value(val, threshold=0.01):
    return '0' if val < threshold else f'{val:.2f}'
