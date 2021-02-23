import os
import subprocess
from optparse import OptionParser

import numpy as np
from numpy import array as npar
import matplotlib.pyplot as plt
import imageio
import glob

import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorflow_docs.vis.embed as embed


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
    plot_model(model, to_file=os.path.join(path,file_name+'_model.png'),show_shapes=True)


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

def generate_and_save_images(model, epoch, test_input,class_label,path,model_name):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = None
    if model_name == 'AC_GAN':
        predictions = model([test_input, tf.convert_to_tensor([[class_label]])], training=False)
    else:
        predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(path, 'image_class_'+str(class_label)+'_at_epoch_{:04d}.png'.format(epoch)))

def create_gif(anim_file,path,class_label):
    with imageio.get_writer(os.path.join(path,anim_file), mode='I') as writer:
        filenames = glob.glob(path+'/image_class_'+str(class_label)+'*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    embed.embed_file(os.path.join(path,anim_file))

@tf.function
def discriminator_loss(cross_entropy,labels,real_output, fake_output):    
    """
    Discriminator Loss
    This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predicitions on real images to an array of 1s
    and the dicriminator's predicitons on fake (generated) images to an array of 0s.
    """
    real_loss = cross_entropy(tf.ones_like(real_output[1]), real_output[1])
    fake_loss = cross_entropy(tf.zeros_like(fake_output[1]), fake_output[1])
    c_loss = cross_entropy(labels,real_output[0]) + cross_entropy(labels,fake_output[0])
    s_loss = real_loss + fake_loss
    total_loss = c_loss + s_loss
    return total_loss

@tf.function
def generator_loss(cross_entropy,labels,fake_output):
    """
    Generator Loss

    The generator's loss quantifies how well it was able to trick the discrimator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1).
    Here, we will compare the discriminators decisions on the generated images to an array of 1s.
    """
    c_loss = cross_entropy(labels,fake_output[0])
    s_loss = cross_entropy(tf.ones_like(fake_output[1]), fake_output[1])
    total_loss = c_loss + s_loss
    return total_loss

def train_step(cross_entropy,generator, discriminator, generator_optimizer, discriminator_optimizer, images, labels ,batch_size,noise_shape):
    """
    The training loop begins with generator receiving a random seed as input. 
    That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). 
    The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.
    """

    # Notice the use of tf.function
    # This annotation causes the function to be "compiled"
    noise_inp = tf.random.normal([batch_size, noise_shape])
    #gen_inp = tf.concat([noise,labels],1)
    label_inp = tf.argmax(labels, axis=1)
    """
    GradientTape() Records operations for automatic differentiation. Operations are recorded if 
    they are executed within this context manager and at least one of their inputs is being "watched".
    """
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise_inp,label_inp], training=True)
        real_output = discriminator(images, training=True) #[0]: cell class (0-3), [1]: fake (0) or real (1)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(cross_entropy,labels, fake_output)
        disc_loss = discriminator_loss(cross_entropy,labels, real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables)) # The zip() function returns an iterator of tuples based on the iterable object.
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss