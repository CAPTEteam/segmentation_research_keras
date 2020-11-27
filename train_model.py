#!/usr/bin/env python3

"""
@author: simadec
"""
from utils.prepare_data import reset_keras, make_train_valid_temp, save_model
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import tensorflow as tf
import xlwt
import segmentation_models as sm
from tensorflow.keras import Model
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.python.keras import backend as K
import albumentations
from ImageDataAugmentor.image_data_augmentor import *
import glob
from pathlib import Path
import shutil
from tensorflow.keras.models import load_model
import GPUtil

reset_keras() #

study_PATH = '/home/capte-gpu-1/Documents/espaces_personnel/SIMON/trash/playground2/'
name_model = 'dev'
my_file = Path("temp_training/")
if my_file.exists():
    shutil.rmtree('temp_training/')

[num_train_images, num_val_images] = make_train_valid_temp(study_PATH)

batch_size=4 # carefull that number images / batch size is less than one
img_height=512
img_width=512
epoch=100 #can be more if you have a big datasets (more than 2000 images)
preprocess_inputEff = sm.get_preprocessing('efficientnetb2')

AUGMENTATIONS = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ElasticTransform(),
])

def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        mask = mask/mask.max()  
        # print('Mask Max -----------------------------------------------------', mask.max())
        # fix, ax = plt.subplots(1,2, figsize=(batch_size*2,10))
        # ax[0].imshow(img[0])
        # ax[1].imshow(mask[0,:,:,0])
        # plt.savefig('aug_visualize3.png')
        yield (preprocess_inputEff(img), mask[:,:,:,0])

train_img_data_gen = ImageDataAugmentor(augment=AUGMENTATIONS, augment_seed=123)
train_img_gen = train_img_data_gen.flow_from_directory('temp_training/Training/images/',target_size=(img_height,img_width),class_mode=None, shuffle=True, seed=123,batch_size=batch_size)
train_mask_data_gen = ImageDataAugmentor(augment=AUGMENTATIONS, augment_seed=123, augment_mode='mask')
train_mask_gen = train_mask_data_gen.flow_from_directory('temp_training/Training/masks/',target_size=(img_height,img_width), class_mode=None, shuffle=True, seed=123,batch_size=batch_size)
train_gen = my_image_mask_generator(train_img_gen,train_mask_gen)

val_img_data_gen = ImageDataAugmentor(augment=AUGMENTATIONS, augment_seed=123)
val_img_gen = val_img_data_gen.flow_from_directory('temp_training/Validation/images/',target_size=(img_height,img_width),class_mode=None, shuffle=True, seed=123,batch_size=batch_size)
val_mask_data_gen = ImageDataAugmentor(augment=AUGMENTATIONS, augment_seed=123, augment_mode='mask')
val_mask_gen = val_mask_data_gen.flow_from_directory('temp_training/Validation/masks/',target_size=(img_height,img_width), class_mode=None, shuffle=True, seed=123,batch_size=batch_size)
val_gen = my_image_mask_generator(val_img_gen,val_mask_gen)

IMG_SIZE=512
# arrêter d'entrainer le modèle quand la mean_iou sur la zone de validation commence à stagner
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=15, mode='min') #ou max ??
# reduce learning rate during the training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4,patience=7, verbose=1,min_lr=0.00005, mode='min')
#optimizeAdam = Adam(lr=0.00075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizeAdam = Adam(lr=0.00075)
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE,3), name='image_in')
unetmodel = sm.Unet(backbone_name='efficientnetb2', encoder_weights='imagenet',input_shape=(IMG_SIZE, IMG_SIZE, 3),classes=1)
out = unetmodel(input_layer)
model = Model(inputs=[input_layer], outputs=out)
#model = multi_gpu_model(model)

lrIni = 0.00075
numLoop=1

for i in range(0,numLoop):
    my_file = Path("temp_training/checkpoint{}.h5".format(i-1))
    if my_file.is_file():
        print("training round {}".format(i+1))
        print("loading checkpoint ... ")
        model = load_model(Path("temp_training/checkpoint{}.h5".format(i-1)))
        print("checkpoint loaded !")
    else:
        print("first training round")
        
    model.compile(optimizer=Adam(lr=lrIni/(10**i)), loss='binary_crossentropy', metrics = ['accuracy'])     
    print("model compiled with new learning rate")
    mc = ModelCheckpoint('temp_training/checkpoint{}.h5'.format(i), period=1, monitor='val_loss',mode='min', save_best_only=True)
    model.fit_generator(train_gen,validation_data = val_gen, steps_per_epoch=num_train_images//batch_size,validation_steps = num_val_images//batch_size, epochs=epoch,callbacks=[reduce_lr,early_stopping_callback,mc])
    reset_keras()

model_final = load_model("temp_training/checkpoint{}.h5".format(numLoop-1))
save_model(model_final,"dev")


 
