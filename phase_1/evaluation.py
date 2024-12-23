import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


class TestModel():
    """
    A PyTorch-based class for testing a model.
    """
    def __init__(self, test_dataset, device, model):
        """
        Args:
            test_dataset (obj): The image dataset for testing.
            device (str): The name of the device to run training/testing.
            model (obj): A PyTorch model.
        """
        self.test_dataset = test_dataset
        self.device = device
        self.model = model
    
    def get_image_index_by_class(self, target_class):
        """
        Loads a checkpoint for a model from a previous training.

        Args:
            target_class (int): A specific class in the image dataset.

        Returns:
            int: The index of a random image belonging to the target class.
        """
        class_indices = [i for i, (_, label) in enumerate(self.test_dataset) if label == target_class]
        if class_indices:
            return random.choice(class_indices)  # Return the index of an image in the specified class
        else:
            return None  # Return None if no image of the specified class is found
    
    def get_confidence_probabilities(self, class_names, target_class):
        """
        This prints the model's confidence for the class an image actually belongs to as well as the class it believes the image belongs to.
        It also plots the model's confidence levels in all classes for that image.

        Args:
            class_names (obj): A list of class names in the image dataset.
            target_class (int): A specific class in the image dataset
        """
        self.model.eval()
        image_index = self.get_image_index_by_class(target_class)
        image, label = self.test_dataset[image_index]
        image = image.to(self.device)
        with torch.no_grad():
            image = image.unsqueeze(0)  # Add batch dimension
            outputs = self.model(image)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence = probabilities[label].item()  # Confidence of the true class
            predicted_class = torch.argmax(probabilities).item()  # Class with the highest probability
            predicted_confidence = probabilities[predicted_class].item()  # Confidence of the predicted class

        print(f"Confidence of True Class '{class_names[label]}': {confidence:.4f}")
        print(f"Confidence of Predicted Class: '{class_names[predicted_class]}': {predicted_confidence:.4f}")

        plt.figure(figsize=(20, 15))
        bars = plt.bar(class_names, probabilities.cpu())  # Move probabilities to CPU
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
        plt.xticks(rotation=90, fontsize=8)
        plt.ylim(0, 1.05) 

        # Add text annotations for probability values above the bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{prob:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)

        plt.tight_layout()
        plt.show()
    
    
class VisualizeModel():
    """
    A PyTorch-based class for visualizing the filters of a model.
    """
    def __init__(self, model):
        """
        Args:
            model (obj): A PyTorch model.
        """
        self.model = model
        self.model_layers, self.model_weights = self.extract_layers_weights()
        
    def extract_layers_weights(self):
        """
        Extracts information about the convolutional layers in a model as well as their weights.

        Returns:
            model_layers (obj): A list of the convolutional layers in a model.
            model_weights (obj): A list of the weights of each layer in a model.
        """
        model_layers = [] # Save the conv layers in this list
        model_weights = [] # Save the conv layer weights in this list

        # Get all the model children as list
        model_children = list(self.model.children())

        # Append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv3d or type(model_children[i]) == nn.Conv2d:
                model_weights.append(model_children[i].weight)
                model_layers.append(model_children[i])

        return model_layers, model_weights

    def display_layers_weights(self):
        """
        Prints information about the convolutional layers in a model as well as their weights.
        """
        # Inspect the conv layers and the respective weights
        for i, (layer, weight) in enumerate(zip(self.model_layers, self.model_weights)):
            # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
            print(f"Layer {i}: {layer} ===> Shape: {weight.shape}")

    def visualize_layer_filters(self, layer_num, layer_shape):
        """
        This plots the filter for a convolutional layer in a model.
        The number of where the convolutional layer is located in the model has to be specified.
        This will be able to plot the filter for either a 3D or 2D convolutional layer.
        
        Args:
            layer_num (int): The number of a convolutional layer in a model.
            layer_shape (str): The shape of a convolutional layer in a model.
        """
        # Get the weights of the specified convolutional layer in the model
        model_layer = self.model_weights[layer_num].data

        # Check if the layer is 3D (e.g., for 3D convolutional layers)
        if layer_shape == '3D':
            # Extract the dimensions of the layer's weight tensor
            n_filters, in_channels, d, h, w = model_layer.shape

            # Set the number of columns for subplots to 4 (can be adjusted)
            ncols = 4

            # Calculate the number of rows required to display all filters
            nrows = int(np.ceil(n_filters / ncols))

            # Create a new figure with subplots and adjust spacing
            fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0.5)

            # Iterate through each subplot
            for i, ax in enumerate(axes.flat):
                ax.set_aspect('equal')
                ax.axis('off')
                # Check if there are more filters to display
                if i < n_filters:
                    # Get the data for the current filter and convert it to a NumPy array
                    filter_data = model_layer[i, 0].cpu().numpy()
                    # Display a central slice of the filter (you can adjust the slice as needed)
                    ax.imshow(filter_data[:, :, 1].T, cmap='gray')

        # Check if the layer is 2D (e.g., for 2D convolutional layers)
        elif layer_shape == '2D':
            # Extract the dimensions of the layer's weight tensor
            n_filters, in_channels, h, w = model_layer.shape

            # Set the number of columns for subplots to 8 (can be adjusted)
            ncols = 8

            # Calculate the number of rows required to display all filters
            nrows = int(np.ceil(n_filters / ncols))

            # Calculate the aspect ratio to maintain filter shape
            aspect_ratio = h / w

            # Create a new figure with subplots and adjust spacing
            fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
            fig.subplots_adjust(hspace=0, wspace=0)

            # Iterate through each subplot
            for i, ax in enumerate(axes.flat):
                ax.set_aspect(aspect_ratio)
                ax.axis('off')
                # Check if there are more filters to display
                if i < n_filters:
                    # Get the data for the current filter and convert it to a NumPy array
                    filter_data = model_layer[i, 0].cpu().numpy()
                    # Display the filter using a grayscale colormap
                    ax.imshow(filter_data, cmap='gray')

        # Display the created subplots
        plt.show()