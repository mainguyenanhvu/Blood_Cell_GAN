import os
import sys
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from training_helper import *
from model_admin import *
from dataset_admin import *

model_path = 'model_files'
history_path = 'history'
extent_name = ''

ACCEPTABLE_AVAILABLE_MEMORY = 1024
OPTIMIZER="Adam"
LOSS="categorical_crossentropy"
METRICS=["accuracy"]
EPOCHS = 100

def train_naive_GAN(gan, discriminator, dataset):
    discriminator.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    discriminator.trainable = False
    gan.compile(loss="binary_crossentropy", optimizer=OPTIMIZER)
    
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
    return history

def main(options,train_dic):
    mkdir_if_missing(options.save_path)
    mkdir_if_missing(os.path.join(options.save_path,model_path))
    mkdir_if_missing(os.path.join(options.save_path,history_path))

    dataset = Dataset(options)
    gan, generator, discriminator = create_model(options.model_name,dataset)
    history = train_dic[options.model_name](gan,discriminator,dataset)

    file_name = options.model_name+extent_name
    model_file = os.path.join(options.save_path,model_path+'/'+file_name+'.hd5')
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=options.verbose, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    generator, discriminator = gan.layers
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
        train_dic = {'naive_GAN': train_naive_GAN}
        main(options,train_dic)