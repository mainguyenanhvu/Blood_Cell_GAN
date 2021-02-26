import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from training_helper import *
from model_admin import *
from dataset_admin import *

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


model_path = 'model_files'
history_path = 'history'
extent_name = ''

ACCEPTABLE_AVAILABLE_MEMORY = 1024
OPTIMIZER="Adam"
LOSS="categorical_crossentropy"
METRICS=["accuracy"]
EPOCHS = 10

def train_naive_GAN(gan, generator, discriminator, dataset, model_name):
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, dataset.noise_shape])
    discriminator.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    
    output_path = '/home/anhkhoa/Vu_working/GAN/Blood_Cell/result/output/vanilla_GAN'
    generator, discriminator = gan.layers
    history = {'epoch':[], 'disc_loss_epoch':[], 'gan_loss_epoch':[]}
    n_batch = dataset.train_generator.samples//dataset.batch_size
    for epoch in tqdm(range(EPOCHS),desc='Epoch'):
        batches = 0  
        batch_bar = tqdm(total=n_batch,desc='Batch ' + str(epoch+1),position=0, leave=True)
        for X_batch, y_batch in dataset.train_generator:
            batch_size = len(X_batch)
            batch_bar.update(1)
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size,dataset.noise_shape])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            disc_loss = discriminator.train_on_batch(X_fake_and_real, y1, return_dict=True)
            
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, dataset.noise_shape])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(noise, y2, return_dict=True)
            batches+=1
            if batches >= n_batch:
                break
        batch_bar.close()
        
        #history['epoch'].append(epoch+1)
        history['disc_loss_epoch'].append(disc_loss['loss'])
        history['gan_loss_epoch'].append(gan_loss['loss'])
        generate_and_save_images(generator, epoch, seed,0,output_path,'vanilla_GAN')
    create_gif('naive_GAN_class_'+'.gif',output_path,0)
    return history

    
def train_AC_GAN(gan, generator, discriminator, dataset, model_name):    
    num_examples_to_generate = 16

    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, dataset.noise_shape])
    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.CategoricalCrossentropy()
    """
    The discriminator and the generator optimizers are different since we will train two networks separately.
    The Adam optimization algorithm is an extension to stochastic gradient descent.
    Stochastic gradient descent maintains a single learning rate (termed alpha) for all weight updates and the learning rate does not change during training.
    A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.
    """
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    #discriminator.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER)
    #discriminator.trainable = False
    #gan.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER)
    
    #generator, discriminator = gan.layers
    output_path = '/home/anhkhoa/Vu_working/GAN/Blood_Cell/result/output/'+model_name
    checkpoint_dir = "/home/anhkhoa/Vu_working/GAN/Blood_Cell/result/model_files/"+model_name
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)
    history = {'epoch':[], 'disc_loss_epoch':[], 'gan_loss_epoch':[]}
    n_batch = dataset.train_generator.samples//dataset.batch_size
    for epoch in tqdm(range(EPOCHS),desc='Epoch',position=0):
        batches = 0  
        batch_bar = tqdm(total=n_batch,desc='Batch ' + str(epoch+1),position=epoch+1, leave=True)
        for X_batch, y_batch in dataset.train_generator:
            batch_size = len(X_batch)
            batch_bar.update(1)
            gan_loss, disc_loss = train_step(cross_entropy, generator, discriminator, generator_optimizer, discriminator_optimizer, X_batch, y_batch ,batch_size,dataset.noise_shape)
            batches+=1
            if batches >= n_batch:
                break
        checkpoint.save(file_prefix = checkpoint_prefix)
        batch_bar.close()
        
        #history['epoch'].append(epoch+1)
        history['disc_loss_epoch'].append(disc_loss)
        history['gan_loss_epoch'].append(gan_loss)
        for i in range(4):
            generate_and_save_images(generator, epoch, seed,i,output_path,'AC_GAN')
    for i in range(4):
        create_gif(model_name + '_class_'+str(i)+'.gif',output_path,i)
    return history

def main(options,train_dic):
    mkdir_if_missing(options.save_path)
    mkdir_if_missing(os.path.join(options.save_path,model_path))
    mkdir_if_missing(os.path.join(options.save_path,history_path))

    dataset = Dataset(options)
    gan, generator, discriminator = create_model(options.model_name,dataset)
    history = train_dic[options.model_name](gan,generator,discriminator,dataset, options.model_name)

    file_name = options.model_name+extent_name
    model_file = os.path.join(options.save_path,model_path+'/'+file_name+'.hd5')
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=options.verbose, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    #generator, discriminator = gan.layers
    generator.save(model_file)

    his_path = os.path.join(options.save_path,history_path)
    draw_model(generator, his_path,file_name + '_generator')
    draw_model(discriminator, his_path,file_name + '_discriminator')
    save_history(history, his_path,file_name)
    plot_stats(history, his_path, file_name)


if __name__ == '__main__':
    options, args = create_training_opt_parser()
    gpu_id = get_gpu_id_max_memory(ACCEPTABLE_AVAILABLE_MEMORY)
    if gpu_id == -1:
        print("Can't run on GPUs now because of lacking available memory!")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        train_dic = {'naive_GAN': train_naive_GAN, 'vanilla_GAN': train_naive_GAN, 'AC_GAN': train_AC_GAN, 
        'DC_GAN': train_AC_GAN}
        main(options,train_dic)