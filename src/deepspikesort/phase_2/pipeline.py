import datetime
import logging
import os
import time

import numpy as np
import h5py

import torch
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

# from cuml import KMeans
# from pycave.bayes import GaussianMixture
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

import spikeinterface.full as si

from . import comparison
from .. import load_dataset
from ..util import AverageMeter, calculate_elapsed_time, print_epoch_header


class DeepSpikeSortPipeline():
    def __init__(self, dataset, batch_size, cnn, loss_fn, optimizer, accelerator, num_units, output_folder, session_id, verbose_count=25):
        """
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset.
        cnn : torch.nn.Module
            The convolutional neural network.
        loss_fn : torch.nn.CrossEntropyLoss
            The loss function.
        optimizer : torch.optim.Adam
            The optimizer.
        accelerator : Accelerator
            The Accelerator instance for handling distributed training.
        session_id : str
            An ID for the training session.
        verbose_count : int, optional
            Number of times to print detailed progress.
        """
        self.dataset = dataset
        self.batch_size = batch_size

        self.cnn = cnn
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accelerator = accelerator

        self.num_units = num_units

        self.output_folder = output_folder
        self.session_id = session_id

        self.verbose_count = verbose_count


    def extract_features(self):
        """
        Extract features from the dataset.

        Returns
        -------
        numpy.ndarray
            Array of preprocessed features.
        """
        # Prepare model and dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        self.cnn, dataloader = self.accelerator.prepare(self.cnn, dataloader)
        self.cnn.eval()

        self.accelerator.wait_for_everyone()

        # Calculate exact features needed for this process
        total_samples = len(self.dataset)
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        samples_per_process = total_samples // world_size
        remaining_samples = total_samples % world_size

        # Calculate exact samples for current process
        local_samples = samples_per_process + (1 if rank < remaining_samples else 0)

        # Pre-allocate features array with exact size needed
        features_local = np.zeros((local_samples, 5000), dtype='float32')
        local_idx = 0

        # Initialize meters and logging
        data_time_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        batches_count = len(dataloader)
        verbose_interval = max(1, batches_count // self.verbose_count)
        padding = len(str(batches_count))

        if self.accelerator.is_main_process:
            print("- Extraction")
            print("Extracting features...")

        extract_end = time.time()
        batch_end = time.time()

        # Extract features
        with torch.no_grad():
            for batch, X in enumerate(dataloader):
                data_time_meter.update(calculate_elapsed_time(batch_end))

                with self.accelerator.autocast():
                    outputs = self.cnn(X, feature_extraction=True)

                # Store features
                batch_size = outputs.shape[0]
                features_local[local_idx:local_idx + batch_size] = outputs.cpu().numpy()
                local_idx += batch_size

                batch_time_meter.update(calculate_elapsed_time(batch_end))
                batch_end = time.time()

                if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                    datetime_formatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                    batch_formatted = f"{batch+1:0{padding}d}"
                    print(f"{datetime_formatted} - [{batch_formatted}/{len(dataloader)}]\t"
                          f"Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t"
                          f"Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})")

        self.accelerator.wait_for_everyone()

        # Gather features from all processes
        features_gathered = self.accelerator.gather(torch.from_numpy(features_local)).cpu().numpy()

        # Clean up local features
        del features_local
        torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            # Concatenate features in correct order
            features_reconstructed = features_gathered[:total_samples]
            del features_gathered

            # Preprocess features
            features_preprocessed = self.preprocess_features(features_reconstructed)
            del features_reconstructed

            print(f"\nExtraction time: {calculate_elapsed_time(extract_end):.3f}\n")

            return features_preprocessed
        else:
            return None


    def preprocess_features(self, features):
        """
        Preprocess features using PCA, whitening, and L2 normalization.

        Parameters
        ----------
        features : numpy.ndarray
            Array of features.

        Returns
        -------
        numpy.ndarray
            Array of preprocessed features.
        """
        # Dimensionality reduction
        features_pca = PCA(n_components=50).fit_transform(features)

        # Add locations x and y of peak channel
        # if self.dataset.method == "slice":
        #     features_sliced = np.hstack((features_reduced, self.dataset.waveform_locs))

        # Whitening
        features_whitened = StandardScaler().fit_transform(features_pca)

        # L2-normalization
        features_normalized = normalize(features_whitened, norm="l2", axis=1)

        return features_normalized


    def cluster_features(self, features):
        """
        Cluster preprocessed features using Gaussian Mixture Model.

        Parameters
        ----------
        features : numpy.ndarray
            Array of features.

        Returns
        -------
        numpy.ndarray
            Array of cluster labels.
        """
        print("Clustering features...")
        cluster_time = time.time()

        labels = GaussianMixture(
            n_components=self.num_units,
            random_state=0
        ).fit_predict(features)

        print(f"\nClustering time: {time.time() - cluster_time:.3f}\n")

        return labels


    def relabel_clusters(self, previous_sorting, current_sorting, cluster_labels):
        """Maintains consistent cluster labeling between epochs."""
        # Compare the two sortings
        comparison = si.compare_two_sorters(
            sorting1=current_sorting,
            sorting2=previous_sorting,
            sorting1_name="Current",
            sorting2_name="Previous",
            delta_time=0,
        )

        # Get the matching between current and previous labels
        matching = comparison.get_matching()[0]

        # Find missing labels from the previous set
        all_possible_labels = set(range(len(matching)))
        used_labels = set(label for label in matching if label != -1)
        missing_labels = list(all_possible_labels - used_labels)

        # Randomly shuffle the missing labels
        np.random.shuffle(missing_labels)

        # Replace -1 values with randomly selected missing labels
        for i in range(len(matching)):
            if matching[i] == -1:
                if missing_labels:
                    matching[i] = missing_labels.pop(0)

        # Create the mapping
        label_map = {i: int(matching[i]) for i in range(len(matching))}

        # Apply the mapping to cluster_labels
        new_labels = np.array([label_map[label] for label in cluster_labels], dtype=int)

        return new_labels


    def train_cnn(self, cluster_labels):
        """
        Trains the convolutional neural network using the cluster labels.

        Parameters
        ----------
        labels : numpy.ndarray
            Cluster labels for the dataset.

        Returns
        -------
        float
            The average loss over the training dataset.
        """
        # Create dataset with pseudo-labels
        dataset = load_dataset.ClusteredDataset(
            self.dataset.dataset_folder,
            self.dataset.trace_inds,
            cluster_labels,
            self.dataset.properties,
            self.dataset.channel_locations,
            self.dataset.method
        )

        # Create dataloader with proper batch size scaling
        # Scale batch size by number of processes for effective global batch size
        local_batch_size = self.batch_size // self.accelerator.num_processes
        local_batch_size = max(1, local_batch_size)  # Ensure at least 1

        dataloader = DataLoader(
            dataset,
            batch_size=local_batch_size,
            num_workers=4,
            pin_memory=True
        )

        # Prepare model, optimizer and dataloader
        self.cnn, self.optimizer, dataloader = self.accelerator.prepare(
            self.cnn, self.optimizer, dataloader
        )
        self.cnn.train()

        self.accelerator.wait_for_everyone()

        # Calculate exact predictions needed for this process
        total_samples = len(dataset)
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        samples_per_process = total_samples // world_size
        remaining_samples = total_samples % world_size
        local_samples = samples_per_process + (1 if rank < remaining_samples else 0)

        # Pre-allocate predictions array for local process
        labels_predicted_local = np.empty(local_samples, dtype="<i8")
        local_idx = 0

        # Initialize meters
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        # Setup logging
        batches_count = len(dataloader)
        verbose_interval = max(1, batches_count // self.verbose_count)
        padding = len(str(batches_count))

        if self.accelerator.is_main_process:
            print("- Training")
            print(f"Training CNN (global batch size: {self.batch_size}, local batch size: {local_batch_size})...")

        train_end = time.time()
        batch_end = time.time()

        for batch, (X, Y) in enumerate(dataloader):
            data_time_meter.update(calculate_elapsed_time(batch_end))

            # Training step
            self.optimizer.zero_grad()
            outputs = self.cnn(X)
            _, preds = torch.max(outputs, 1)

            # Store predictions
            batch_size = preds.shape[0]
            labels_predicted_local[local_idx:local_idx + batch_size] = preds.cpu().numpy()
            local_idx += batch_size

            # Loss computation
            Y = Y.long()
            loss = self.loss_fn(outputs, Y)
            loss_meter.update(loss.item(), X.shape[0])

            # Accuracy computation
            correct_preds = (preds == Y).sum().item()
            accuracy_meter.update(correct_preds / Y.size(0), X.size(0))

            # Backward pass
            self.accelerator.backward(loss)
            self.optimizer.step()

            batch_time_meter.update(calculate_elapsed_time(batch_end))
            batch_end = time.time()

            # Logging
            if batch % verbose_interval == 0 and self.accelerator.is_main_process:
                datetime_formatted = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                batch_formatted = f"{batch+1:0{padding}d}"
                print(f"{datetime_formatted} - [{batch_formatted}/{len(dataloader)}]\t"
                      f"Time: {batch_time_meter.val:.3f} ({batch_time_meter.avg:.3f})\t"
                      f"Data: {data_time_meter.val:.3f} ({data_time_meter.avg:.3f})\t"
                      f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                      f"Accuracy: {accuracy_meter.val:.4f} ({accuracy_meter.avg:.4f})")

        self.accelerator.wait_for_everyone()

        # Gather predictions from all processes
        labels_predicted_tensor = torch.from_numpy(labels_predicted_local)
        labels_gathered = self.accelerator.gather(labels_predicted_tensor)

        # Gather metrics from all processes
        loss_avg = self.accelerator.gather(torch.tensor(loss_meter.avg, device=self.accelerator.device)).mean()
        accuracy_avg = self.accelerator.gather(torch.tensor(accuracy_meter.avg, device=self.accelerator.device)).mean()

        if self.accelerator.is_main_process:
            print(f"\nTraining time: {calculate_elapsed_time(train_end):.3f}")
            # Take only valid samples and convert to numpy
            labels_predicted = labels_gathered.cpu().numpy()[:total_samples]
            return labels_predicted, loss_avg.item(), accuracy_avg.item()
        else:
            return None, None, None


    def run_deepspikesort(self, num_epochs):
        # Initialize handlers and loggers
        output_data_handler = None
        if self.accelerator.is_main_process:
            output_data_handler = self.setup_output_data_handler(num_epochs)

        performance_metrics_logger = self.setup_performance_metrics_logger()

        # Ensure all processes are synchronized before starting
        self.accelerator.wait_for_everyone()

        total_end = time.time()

        # Store previous data as None initially
        previous_sorting = None
        previous_cluster_labels = None
        cluster_labels = None

        for epoch in range(num_epochs):
            try:
                if self.accelerator.is_main_process:
                    print_epoch_header(epoch)

                # Feature extraction and clustering every 5 epochs
                if epoch % 5 == 0:
                    # Step 1: Feature Extraction (already distributed)
                    features = self.extract_features()

                    # Step 2: Clustering
                    if self.accelerator.is_main_process:
                        # Perform clustering
                        cluster_labels = self.cluster_features(features)

                        # Handle cluster relabeling if not first epoch
                        if epoch != 0:
                            current_sorting = comparison.create_numpy_sorting(
                                self.dataset.properties["sample_index"],
                                cluster_labels,
                                30000
                            )
                            cluster_labels = self.relabel_clusters(
                                previous_sorting,
                                current_sorting,
                                cluster_labels
                            )

                        # Update previous data
                        previous_cluster_labels = cluster_labels
                        previous_sorting = comparison.create_numpy_sorting(
                            self.dataset.properties["sample_index"],
                            previous_cluster_labels,
                            30000
                        )

                # Ensure all processes have cluster labels
                self.accelerator.wait_for_everyone()

                # Step 3: Training
                labels_predicted, loss, accuracy = self.train_cnn(cluster_labels)

                # Handle metrics and logging
                if self.accelerator.is_main_process:
                    performance_metrics = np.array(
                        (loss, accuracy),
                        dtype=output_data_handler["metrics"].dtype
                    )

                    # Log and save data
                    self.log_performance_metrics(
                        performance_metrics_logger,
                        epoch,
                        performance_metrics
                    )
                    self.save_output_data(
                        output_data_handler,
                        epoch,
                        features,
                        labels_predicted,
                        performance_metrics
                    )

                # Synchronize before next epoch
                self.accelerator.wait_for_everyone()

            except KeyboardInterrupt:
                break

        # Cleanup and final steps
        if self.accelerator.is_main_process:
            self.plot_performance_metrics(output_data_handler)
            output_data_handler.close()
            print(f"\nTotal time: {calculate_elapsed_time(total_end):.3f}")

        # Final synchronization
        self.accelerator.wait_for_everyone()


    def setup_output_data_handler(self, num_epochs):
        """
        Sets up an HDF5 file to store DeepSpikeSort output data.

        Parameters
        ----------
        num_epochs : int
            Number of epochs.

        Returns
        -------
        h5py.File
            An HDF5 file handle with datasets.
        """
        file = os.path.join(self.output_folder, f"{self.session_id}_output_data.h5")
        handler = h5py.File(file, "w")

        handler.create_dataset("properties", data=self.dataset.properties)

        handler.create_dataset("features", shape=(num_epochs, len(self.dataset), 50), dtype="<f8")
        handler.create_dataset("labels", shape=(num_epochs, len(self.dataset)), dtype="<i8")

        handler.create_dataset("metrics", shape=(num_epochs,),
                               dtype=np.dtype([("loss", "<f8"),
                                               ("accuracy", "<f8")]))

        return handler


    def save_output_data(self, handler, epoch, features, labels, metrics):
        """
        Saves DeepSpikeSort output data to an HDF5 file.

        Parameters
        ----------
        handle : h5py.File
            The HDF5 file handle with datasets.
        epoch : int
            The epoch number.
        features : numpy.ndarray
            Array of features.
        labels : numpy.ndarray
            Array of labels.
        """
        # Store features and labels for the current epoch
        handler["features"][epoch, :, :] = features
        handler["labels"][epoch, :] = labels
        handler["metrics"][epoch] = metrics


    def setup_performance_metrics_logger(self):
        """
        Sets up a logger to record DeepSpikeSort performance metrics.

        Returns
        -------
        logging.Logger
            A logger object to record performance metrics.
        """
        file = os.path.join(self.output_folder, f"{self.session_id}_performance_metrics.log")

        # Initialize a logger
        logger = logging.getLogger("performance_metrics_logger")
        logger.setLevel(logging.INFO)

        # Create a file handler to write logs to the file
        handler = logging.FileHandler(file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        # Set the formatter for the file handler
        handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(handler)

        return logger


    def log_performance_metrics(self, logger, epoch, metrics):
        """
        Logs DeepSpikeSort performance metrics to a log file.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        logger.info("[{0:03}]\t"
                    "Loss: {1:.4f}\t"
                    "Accuracy: {2:.4f}"
                    .format(epoch,
                            metrics["loss"],
                            metrics["accuracy"]))


    def plot_performance_metrics(self, handle):
        """
        Plots DeepSpikeSort performance metrics.

        Parameters
        ----------
        metrics_progress : list of np.array
            List where each np.array contains values for loss, silhouette score, Davies-Bouldin index, and Calinski-Harabasz index for each epoch.
        """
        metrics = handle["metrics"][:]

        epochs = range(metrics.shape[0])

        plt.figure(figsize=(10, 10))

        # loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, metrics["loss"], "b", label="Loss")
        plt.title("Loss")

        # Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, metrics["accuracy"], "g", label="Accuracy")
        plt.title("Accuracy")

        plt.figtext(0.5, 0.04, "Epochs", ha="center", va="center", fontsize=11)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.savefig(os.path.join(self.output_folder, f"{self.session_id}_performance_metrics.png"))
