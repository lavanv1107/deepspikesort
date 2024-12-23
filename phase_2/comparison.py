import os
import sys

import numpy as np

import spikeinterface.full as si

import logging

sys.path.append("..")
import util


def log_si_comparison(sorting1, sorting2, sorting1_name, sorting2_name, num_units, peak_units, dss_labels, results_folder):
    cmp_dss_peaks = si.compare_two_sorters(
        sorting1=sorting1,
        sorting2=sorting2,
        sorting1_name=sorting1_name,
        sorting2_name=sorting2_name,
        verbose=True
    )
    
    agreement_matrix = (si.plot_agreement_matrix(cmp_dss_peaks)).figure
    agreement_matrix.savefig(os.path.join(results_folder, f'{num_units:0>3d}_agreement_matrix.png'))
    
    dss_to_peaks, _ = cmp_dss_peaks.get_matching()
    
    comparison_results_file = os.path.join(results_folder, f'{num_units:0>3d}_comparison_results.log')    
    logger_cmp = logging.getLogger('logger_cmp')
    logger_cmp.setLevel(logging.INFO)
    file_handler_cmp = logging.FileHandler(comparison_results_file, mode='w')
    formatter_cmp = logging.Formatter('%(message)s')
    file_handler_cmp.setFormatter(formatter_cmp)
    logger_cmp.addHandler(file_handler_cmp)
         
    logger_cmp.info('Dataset: {0}\n{6}'
                    'Peak units: \n{1}\n{6}\n'
                    'DSS labels: \n{2}\n{6}\n'
                    'Matched units:\n{3}\n{6}\n'
                    'Match event count:\n{4}\n{6}\n'
                    'Agreement scores:\n{5}'
                    .format(len(dss_labels), util.format_labels(peak_units), util.format_labels(dss_labels),
                            dss_to_peaks, cmp_dss_peaks.match_event_count, cmp_dss_peaks.agreement_scores,
                            util.write_separator()))
    
    
def create_numpy_sorting(times, cluster_labels, sampling_frequency):        
    times, cluster_labels = sort_times_labels(times, cluster_labels)
    
    numpy_sorting = si.NumpySorting.from_times_labels(times, cluster_labels, sampling_frequency)
    
    return numpy_sorting


def filter_samples_duplicate(spike_train):
    # Find unique values and their indices
    _, indices = np.unique(spike_train['sample_index'], return_index=True)

    # Filter the array to keep only the first occurrences of duplicates and all unique values
    spike_train_filtered = spike_train[indices]
    
    return spike_train_filtered


def sort_times_labels(times, cluster_labels):        
    # Get sort order indices
    indices_sorted = np.argsort(times)

    # Use indices to sort arrays
    times = times[indices_sorted]
    cluster_labels = cluster_labels[indices_sorted]
    
    return times, cluster_labels


def get_unit_scores(scores_dict, unit_id):
    # Define the data type for the structured array
    dtype = np.dtype([('unit_id', 'i8'), ('score', 'U20')])

    # Create an empty structured array
    unit_scores = np.zeros(len(scores_dict[unit_id][0]), dtype=dtype)

    # Fill the structured array with data
    unit_scores['unit_id'] = unit_id
    unit_scores['score'] = scores_dict[unit_id][0]

    return unit_scores


def get_scores_rows(scores_dict, unit_ids):
    scores_rows = 0
    for unit_id in unit_ids:
        scores_rows += len(scores_dict[unit_id][0])  
    return scores_rows


def get_scores(scores_dict, unit_ids):
    # Calculate the total number of rows needed in the scores array
    scores_rows = get_scores_rows(scores_dict, unit_ids)

    # Define the data type for the structured array, including the new 'id' field
    dtype = np.dtype([('id', 'i8'), ('unit_index', 'i8'), ('score', 'U20')])

    # Create an empty structured array
    scores = np.zeros(scores_rows, dtype=dtype)

    start_index = 0
    for unit_id in unit_ids:
        unit_scores = get_unit_scores(scores_dict, unit_id)

        # Determine the number of rows in unit_scores
        end_index = start_index + len(unit_scores)

        # Place unit_scores in the scores array
        scores['unit_index'][start_index:end_index] = unit_id
        scores['score'][start_index:end_index] = unit_scores['score']

        # Update the start index for the next unit_scores
        start_index = end_index

    # Fill in the 'id' column with a range of indices
    scores['id'] = np.arange(scores_rows)

    return scores


def get_unmatched_indices(scores):
    mask = scores['score'] != "TP"  

    unmatched_indices = scores[mask]['id']

    return unmatched_indices


def filter_samples_on_match(scores_dict, unit_ids, spike_train):
    scores = get_scores(scores_dict, unit_ids)
    
    unmatched_indices = get_unmatched_indices(scores)
    
    spike_train = np.sort(spike_train, order=['unit_index', 'sample_index'])
    
    spike_train_unmatched = spike_train[unmatched_indices]

    mask = ~np.isin(np.arange(len(spike_train)), unmatched_indices)
    
    spike_train_matched = spike_train[mask]
    
    return spike_train_matched, spike_train_unmatched
    