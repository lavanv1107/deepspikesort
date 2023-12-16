import os

import numpy as np

import spikeinterface.full as si

import logging

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
    agreement_matrix.savefig(os.path.join(results_folder, f'agreement_matrix_{num_units:0>3d}.png'))
    
    dss_to_peaks, _ = cmp_dss_peaks.get_matching()
    
    comparison_results_file = os.path.join(results_folder, f'comparison_results_{num_units:0>3d}.log')    
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


def sort_times_labels(times, cluster_labels):        
    # Get sort order indices
    sort_indices = np.argsort(times)

    # Use indices to sort arrays
    times = times[sort_indices]
    cluster_labels = cluster_labels[sort_indices]
    
    return times, cluster_labels
