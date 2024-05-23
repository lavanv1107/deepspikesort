import datetime
import logging
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.append("..")
from util import AverageMeter, calculate_elapsed_time, print_epoch_header


class ClassifyPipeline():
    """
    A PyTorch-based class for training and validating a convolutional neural network model. 
    It utilizes HuggingFace's accelerator for distributed training.
    """
    def __init__(self, dataset, cnn, loss_fn, optimizer, accelerator, output_folder, session_id, verbose_count=50):
        """
        Initializes the training and validation setup.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset.
        cnn : torch.nn.Module
            The neural network model.
        loss_fn : torch.nn.CrossEntropyLoss
            The loss function.
        optimizer : torch.optim.Adam
            The optimizer.
        accelerator : Accelerator
            The Accelerator instance for handling distributed training.
        output_folder : str
            The folder path for saving output.
        session_id : str
            An ID for the training session.
        verbose_count : int, optional
            Number of times to print detailed progress, by default 50.

        Attributes
        ----------    
        train_dataset : torch.utils.data.Subset
            The training dataset.
        val_dataset : torch.utils.data.Subset
            The validation dataset.
        progress_logger : logging.Logger
            The logger object for logging progress.
        """
        self.dataset = dataset
        self.train_dataset, self.val_dataset = self.split_dataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=256)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=256)
        
        self.cnn = cnn
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accelerator = accelerator
        
        self.output_folder = output_folder
        self.session_id = session_id
        self.progress_logger = self.setup_logger()

        self.verbose_count = verbose_count

    def split_dataset(self):
        """
        Splits the dataset into training and validation.

        Returns
        -------
        tuple of torch.utils.data.Subset
            The training and validation datasets.
        """
        dataset_size = len(self.dataset)
        train_size = int(0.7 * dataset_size)
        val_size = dataset_size - train_size  
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        return train_dataset, val_dataset

    def train(self):
        """
        Trains the model.

        Returns
        -------
        float
            The average loss over the training dataset.
        """
        # Prepare the model, optimizer, and dataloader with the accelerator
        self.cnn, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.cnn, self.optimizer, self.train_dataloader
        )

        # Initialize meters for tracking time and loss
        loss_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        # Total number of batches and formatting for logs
        total_batches = len(self.train_dataloader)
        padding = len(str(total_batches))
        verbose_interval = max(1, total_batches // self.verbose_count)       

        # Set the model to training mode
        self.cnn.train()
        
        if self.accelerator.is_main_process:
            print("- Training")
            print('Training model...')
            
        train_end = time.time()
        batch_end = time.time() 

        # Iterate over each batch
        for batch, (X, Y) in enumerate(self.train_dataloader):
            # Measure data loading time
            data_time_meter.update(calculate_elapsed_time(batch_end))

            # Zero out the gradients before forward pass
            self.optimizer.zero_grad()

            # Forward pass
            inputs_dict = {'x': X}
            pred = self.cnn(inputs_dict)

            # Compute loss
            loss = self.loss_fn(pred, Y)
            loss_meter.update(loss.item(), X.size(0))

            # Backward pass and update model parameters
            self.accelerator.backward(loss)
            self.optimizer.step()

            # Update batch processing time
            batch_time_meter.update(calculate_elapsed_time(batch_end))
            batch_end = time.time() 
            
            # Format batch processing time for logs
            formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

            # Periodically log training progress
            if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                formatted_batch = f"{batch+1:0{padding}d}"
                print(f'{formatted_time} - [{formatted_batch}/{total_batches}]\t'
                      f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                      f'Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t'
                      f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})')

        if self.accelerator.is_main_process:
            print(f'\nTraining time: {calculate_elapsed_time(train_end):.3f}\n')
                
        return loss_meter.avg

        
    def validate(self):
        """
        Validates the model.

        Returns
        -------
        float
            The average loss over the validation dataset.
        float
            The accuracy of the model on the validation dataset.
        list
            The list of true labels from the validation dataset.
        list
            The list of predicted labels from the validation dataset.
        """
        # Prepare the model and dataloader with the accelerator 
        self.cnn, self.val_dataloader = self.accelerator.prepare(
            self.cnn, self.val_dataloader
        )

        # Initialize meters for monitoring loss and time
        loss_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        # Calculate total number of batches and logging interval
        total_batches = len(self.val_dataloader)
        padding = len(str(total_batches))
        verbose_interval = max(1, total_batches // self.verbose_count)

        # Lists to store true and predicted labels
        true_labels = []
        predicted_labels = []

        # Set the model to evaluation mode
        self.cnn.eval()        

        if self.accelerator.is_main_process:
            print("- Validation")
            print('Validating model...')
                    
        val_end = time.time()
        batch_end = time.time()        

        # Iterate over each batch in the validation dataloader
        with torch.no_grad():
            for batch, (X, Y) in enumerate(self.val_dataloader):
                # Measure data loading time
                data_time_meter.update(calculate_elapsed_time(batch_end))

                # Forward pass 
                inputs_dict = {'x': X}
                pred = self.cnn(inputs_dict)

                # Compute loss
                loss = self.loss_fn(pred, Y)
                loss_meter.update(loss.item(), X.size(0))

                # Store true and predicted labels for accuracy computation
                true_labels.extend(Y.tolist())
                predicted_labels.extend(pred.argmax(1).tolist())

                # Update batch processing time
                batch_time_meter.update(calculate_elapsed_time(batch_end))
                batch_end = time.time() 

                # Format batch processing time for logs
                formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

                # Log validation progress at specified intervals
                if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                    formatted_batch = f"{batch+1:0{padding}d}"
                    print(f'{formatted_time} - [{formatted_batch}/{total_batches}]\t'
                          f'Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t'
                          f'Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t'
                          f'Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})')

        # Compute overall accuracy
        correct_predictions = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
        accuracy = correct_predictions / len(true_labels) if true_labels else 0
        
        if self.accelerator.is_main_process:
            print(f'\nValidation time: {calculate_elapsed_time(val_end):.3f}')
                
        return loss_meter.avg, accuracy, true_labels, predicted_labels    


    def train_validate(self, num_epochs):
        """
        Trains and validates a model over a specified number of epochs.

        This method handles the training and validation of a PyTorch model, 
        including checkpointing and early stopping. It logs the progress and metrics
        and visualizes the results after completion.

        Parameters
        ----------
        num_epochs : int
            The number of epochs to train the model.
        """ 
        patience_limit = 10
        improvement_threshold = 0.01
        
        # Load checkpoint if exists
        start_epoch, best_loss, patience_counter, val_losses, val_accs = self.load_checkpoint()

        total_end = time.time()  
        
        for epoch in range(start_epoch + 1, num_epochs + 1):  
            if self.accelerator.is_main_process:
                print_epoch_header(epoch)     

            # Train the model
            self.train()

            # Validate the model
            val_loss, val_acc, true_labels, predicted_labels = self.validate()
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Check for improvement
            if val_loss < best_loss * (1 - improvement_threshold):
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
 
            if self.accelerator.is_main_process:
                # Save checkpoint
                self.save_checkpoint(epoch, best_loss, patience_counter, val_losses, val_accs)
                
                # Log progress
                self.log_progress(epoch, val_loss, val_acc)    
                
                # Plot progress
                self.plot_progress(val_losses, val_accs)

            # Early stopping condition
            if patience_counter >= patience_limit:
                break                
         
        if self.accelerator.is_main_process:
            print(f"\nTotal time: {calculate_elapsed_time(total_end):.3f}")
        
            self.plot_confusion_matrix(true_labels, predicted_labels)    
    
    
    def save_checkpoint(self, epoch, best_loss, patience_counter, losses, accs):
        """
        Saves the current model state as a checkpoint.

        Parameters
        ----------
        folder : str
            The folder path for saving the checkpoint.
        epoch : int
            The current epoch number.
        best_loss : float
            The best validation loss achieved so far.
        patience_counter : int
            The current count for early stopping.
        losses : list
            List of loss values for each epoch.
        accs : list 
            List of accuracy values for each epoch.
        """
        file = os.path.join(self.output_folder, f'{self.session_id}_checkpoint.pth.tar')

        checkpoint = {
            'epoch': epoch,
            'state_dict': self.cnn.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'losses': losses,
            'accs': accs,
        }

        torch.save(checkpoint, file)
        

    def load_checkpoint(self):
        """
        Loads a model state and optimizer state from a checkpoint file.

        Returns
        -------
        tuple
            A tuple containing:
            - epoch (int): The last completed epoch.
            - best_loss (float): The best loss achieved so far.
            - patience_counter (int): The current count for early stopping.
        """
        file = os.path.join(self.output_folder, f'{self.session_id}_checkpoint.pth.tar')
        
        # Check if checkpoint file exists
        if os.path.exists(file):
            # Load checkpoint data
            checkpoint = torch.load(file)

            # Load states into the model and optimizer
            self.cnn.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            # Extract training parameters
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            patience_counter = checkpoint['patience_counter']
            losses = checkpoint['losses']
            accs = checkpoint['accs']

            return epoch, best_loss, patience_counter, losses, accs
        else:
            # Return default values if file doesn't exist
            return 0, float('inf'), 0, [], []
   

    def setup_logger(self):
        """
        Sets up a logger for the model's progress.

        Parameters
        ----------
        progress_file : str
            The file path for logging progress.

        Returns
        -------
        logging.Logger
            The logger object for logging progress.
        """        
        file = os.path.join(self.output_folder, f'{self.session_id}_progress.log')   
        
        # Initialize a logger
        logger = logging.getLogger('progress_logger')
        logger.setLevel(logging.INFO)

        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(file, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(message)s')

        # Set the formatter for the file handler
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)
        
        return logger


    def log_progress(self, epoch, loss, accuracy):       
        """
        Logs the progress of classification.

        Parameters
        ----------        
        epoch : int
            The current epoch number.
        loss : float
            The loss for the current epoch.
        accuracy : float
            The accuracy for the current epoch.
        """
        self.progress_logger.info('[{0:03}]\t'
                                  'Loss: {1:.4f}\t'
                                  'Accuracy: {2:.4f}'
                                  .format(epoch, loss, accuracy))
        
        
    def plot_progress(self, losses, accs):
        """
        Plots the validation loss and accuracy over each epoch.

        Parameters
        ----------  
            losses : list
                List of loss values for each epoch.
            accuracies : list 
                List of accuracy values for each epoch.
        """
        epochs = range(1, len(losses) + 1)

        plt.figure(figsize=(10, 10))

        # Subplot for validation loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, losses, 'b', label='Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')

        # Subplot for validation accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, accs, 'r', label='Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')

        # Save the progress plot to the folder
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, f'{self.session_id}_progress_plot.png'))

        
        
    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """
        Plots a confusion matrix based on true and predicted labels.

        The function converts numerical labels to class names using a provided mapping,
        then generates and saves a confusion matrix as an image in the specified folder.

        Parameters
        ----------
        true_labels : list
            List of true numerical labels.
        predicted_labels : list
            List of predicted numerical labels.
        """
        # Convert encoded labels to original labels
        true_labels = [self.dataset.labels_map[label] for label in true_labels]
        predicted_labels = [self.dataset.labels_map[label] for label in predicted_labels]

        # Sorted list of labels    
        labels = sorted(self.dataset.labels_map.values())

        # Generate a normalized confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=labels, normalize='true')

        # Format the confusion matrix values for display
        formatted_cm = np.array([[format_value(value) for value in row] for row in cm])

        # Determine figure size dynamically based on the number of labels
        num_labels = len(labels)
        dynamic_size = max(7.5, num_labels * 0.75)

        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(dynamic_size, dynamic_size))
        sns.heatmap(cm, annot=formatted_cm, fmt="s", cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Save the confusion matrix plot to the folder
        plt.savefig(os.path.join(self.output_folder, f'{self.session_id}_confusion_matrix.png'))


def format_value(val, threshold=0.01):
    return '0' if val < threshold else f'{val:.2f}'
