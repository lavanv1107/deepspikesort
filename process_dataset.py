import os
import psutil

import numpy as np

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm.notebook import tqdm

import preprocessing


def process_trace(recording, folder, frame):
    """
    Creates a trace of a specified frame and saves it to disk.
 
    Args:
        recording (obj): A RecordingExtractor object created from an NWB file using SpikeInterface.
        folder (str): A folder path name.
        frame (int): A frame number.
    """
    trace_reshaped = preprocessing.get_trace_reshaped(recording, frame)
    trace_file = os.path.join(folder, f"frame_{frame:07d}.npy")  # Zero padding for ordering
    np.save(trace_file, trace_reshaped, allow_pickle=False)


def display_resources():
    nthreads = psutil.cpu_count(logical=True)
    ncores = psutil.cpu_count(logical=False)
    nthreads_per_core = nthreads // ncores
    nthreads_available = len(os.sched_getaffinity(0))
    ncores_available = nthreads_available // nthreads_per_core

    assert nthreads == os.cpu_count()
    assert nthreads == mp.cpu_count()

    print(f'{nthreads=}')
    print(f'{ncores=}')
    print(f'{nthreads_per_core=}')
    print(f'{nthreads_available=}')
    print(f'{ncores_available=}')


def process_unit(table, folder, unit_id, num_processes=128, batch_size=1000, all_frames=False, num_frames=0):
    """
    This creates traces for a specified unit's frames and saves them to disk.
    It creates a folder with a unit's id number. Traces will be created for all frames belonging to that unit and saved to the folder path.
    
    In order to speed up this I/O process, we can utilize multiprocessing as well as batches. 
    The number of processes can be set according to the number of cores available. 
    The batch size can be set according to how much memory is available.
    
    You can either use all frames belonging to a unit or you can set the number of frames to be used.
 
    Args:
        table (obj): A table containing entries with a unit id and frame.
        folder (str): A folder path name.
        unit_id (int): A spike unit's ID number.
        num_processes (int): number of processes for multiprocessing.
        batch_size (int): number of traces to process per batch.
        all_frames (bool): condition to use all frames belonging to a unit.
        num_frames (int): number of frames to use for a unit.
    """
    with mp.Pool(processes=num_processes) as pool:
        unit_folder = os.path.join(folder, f'unit_{unit_id}')
        if not os.path.exists(unit_folder):
            os.mkdir(unit_folder)

        unit_table = get_spike_unit(table, unit_id)
        unit_frames = unit_table.iloc[:, 1].to_list()

        if all_frames:
            num_frames = len(unit_table)

        for i in range(0, num_frames, batch_size):
            frames_batch = unit_frames[i:i+batch_size]
            folder_frames_batch = [(unit_folder, frame) for frame in frames_batch]
            pool.starmap(process_trace, tqdm(folder_frames_batch,
                                             total=len(folder_frames_batch),
                                             desc='processing batch',
                                             dynamic_ncols=True))