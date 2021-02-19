import os
import subprocess
from optparse import OptionParser

import numpy as np
from numpy import array as npar
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model


def create_training_opt_parser():
    parser = OptionParser()
    parser.add_option("-d", "--dataset_path",type=str, default='',
                      help="Dataset path to load (csv file)")
    parser.add_option("-s", "--save_path",type=str, default='./result',
                      help="Save path")

    parser.add_option("-n", "--model_name",type=str, default='naive_GAN',
                      help="Available model to train")
    parser.add_option("--version_run",type=str, default='',
                      help="run th")                      

    parser.add_option("--batch_size", type=int, default=64,
                      help="batch size training")
    parser.add_option("--noise_shape", type=int, default=100,
                      help="noise shape")
    parser.add_option("-v", "--verbose", default=True,
                      help="don't print status messages to stdout")
    (options, args) = parser.parse_args()
    return options, args


def get_gpu_id_max_memory(acceptable_available_memory):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = npar([int(x.split()[0]) for i, x in enumerate(memory_free_info)])
    print(memory_free_values)
    gpu_id = memory_free_values.argmax()
    if memory_free_values[gpu_id] < acceptable_available_memory:
        return -1
    print(gpu_id)
    return gpu_id

def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.mkdir(path)

def draw_model(model,path,file_name):
    plot_model(model, to_file=os.path.join(path,file_name+'_model.png'))


def save_history(history, path, file_name):
    numpy_loss_history = np.array(history['disc_loss_epoch'])
    numpy_loss_history = np.c_[
        np.arange(1, numpy_loss_history.size+1).astype(np.int64), numpy_loss_history]
    numpy_loss_history = np.c_[numpy_loss_history,
                               np.array(history['gan_loss_epoch'])]
    np.savetxt(os.path.join(path, file_name+"loss_history.txt"),
               numpy_loss_history, delimiter=",", header="epoch,disc_loss_epoch,gan_loss_epoch")

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Plot learning curve 
def plot_stats(history, path, file_name, x_label='Epochs', stats='loss'):
    stats, x_label = stats.title(), x_label.title()
    #training_steps = len(history['epoch'])
    plt.figure()
    plt.ylabel(stats)
    plt.xlabel(x_label)
    plt.plot(history['gan_loss_epoch'], label='Training Gan Loss')
    plt.plot(history['disc_loss_epoch'], label='Training Disc Loss')
    # plt.plot(np.linspace(0, training_steps, test_steps), val_stats, label='Validation ' + stats)
    plt.ylim([0,max(plt.ylim())])
    plt.legend(loc='upper right')
    plt.title(file_name)
    plt.savefig(os.path.join(path, file_name+'_hist.png'),bbox_inches='tight')