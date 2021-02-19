from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Dataset():
    def __init__(self,opt):
        self.train_data_path = '/home/anhkhoa/Vu_working/GAN/Blood_Cell/data/dataset2-master/dataset2-master/images/train/train'
        self.test_data_path = '/home/anhkhoa/Vu_working/GAN/Blood_Cell/data/dataset2-master/dataset2-master/images/TEST'
        self.val_data_path = '/home/anhkhoa/Vu_working/GAN/Blood_Cell/data/dataset2-master/dataset2-master/images/train/val'

        self.train_generator = None
        self.test_generator = None
        self.val_test_generator = None
        self.batch_size = opt.batch_size

        self.ndim = 0
        
        self.img_width = 150
        self.img_height =  150
        self.channels = 3
        self.classnum = 4
        self.shape = (self.img_width,self.img_height,self.channels)
        self.create_train_test_test()
        self.noise_shape = opt.noise_shape

    def create_train_test_test(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.test_generator = test_datagen.flow_from_directory(
            self.test_data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.validation_generator = val_datagen.flow_from_directory(
            self.val_data_path,
            target_size=(self.img_height, self.img_width),
            batch_size=1,
            shuffle = False,
            class_mode=None)