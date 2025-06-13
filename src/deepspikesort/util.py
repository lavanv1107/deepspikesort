import time
from collections import Counter, OrderedDict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def print_separator(width=100):
    print('\n' + '-' * width + '\n')
    

def print_epoch_header(epoch):
    """
    Prints the header for each epoch.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    """
    print_separator()
    print(f"Epoch {epoch}")  
    print_separator()
    

def calculate_elapsed_time(start_time):
    return time.time() - start_time

    
def format_value_counts(values, target_columns=10):
    values_counter = OrderedDict(sorted(Counter(values).items()))
    total_items = len(values_counter)
    rows = -(-total_items // target_columns)  # Ceiling division to ensure enough rows
    iterator = iter(values_counter.items())
    formatted_string = ""
    max_key_len = max(len(str(key)) for key in values_counter.keys())
    max_val_len = max(len(str(value)) for value in values_counter.values())

    # Create a matrix (list of lists) to hold the formatted data
    formatted_matrix = [[' ' * (max_key_len + max_val_len + 2) for _ in range(target_columns)] for _ in range(rows)]

    # Fill the matrix column-wise
    for i, (key, value) in enumerate(iterator):
        col = i // rows
        row = i % rows
        formatted_pair = f"{key:0{max_key_len}}: {value:<{max_val_len}}"
        formatted_matrix[row][col] = formatted_pair

    # Convert the matrix to a formatted string
    for row in formatted_matrix:
        formatted_string += '\t'.join(row) + '\n'

    return formatted_string.strip()
