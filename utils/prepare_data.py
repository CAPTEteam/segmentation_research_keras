# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 17:27:43 2018

@author: simadec
"""
from numpy.random import seed
import tensorflow
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
from skimage.io import imread
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gc
#from tensorflow.compat.v1.keras import backend as K
#K.set_session()
from tensorflow.python.keras import backend as K
from pathlib import Path
import shutil
import random
from tensorflow.keras.models import model_from_json
# from keras.backend.tensorflow_backend import set_session
# from keras.backend.tensorflow_backend import clear_session
# from keras.backend.tensorflow_backend import get_session

def reset_keras():
    sess = K.get_session()
    K.clear_session()
    sess.close()
    sess = K.get_session()
    seed(123)
    tensorflow.random.set_seed(1)
    np.random.seed(123)
    SEED = 123
    os.environ['PYTHONHASHSEED']= str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    try:
        del model # this is from global space - change this as you need
    except:
        pass
    try:
        del history # this is from global space - change this as you need
    except:
        pass
    print('gc.collect() : ')
    print(gc.collect()) # if it's done something you should see a number being outputted
    # use the same config as you used to create the session
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 1
    # config.gpu_options.visible_device_list = "0"
    # config.gpu_options.allow_growth = True
    # K.set_session(tf.Session(config=config))
    tf.keras.backend.clear_session()
    print("reset keras")

def make_train_valid_temp(study_PATH,valid =0.1): 
    img_list = []
    train_ids = next(os.walk(study_PATH))[1]
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        images_add = [image for image in Path(study_PATH + id_ + '/').glob("images/*") if image.name.endswith((".png", ".jpg", ".JPG"))]
        img_list = img_list+images_add
    random.shuffle(img_list)
    train_list = img_list[:int(len(img_list)*(1-valid))]
    valid_list = img_list[int(len(img_list)*(1-valid)):]

    processed_train_img = Path('temp_training/Training/images/img')
    processed_train_img.mkdir(parents=True, exist_ok=True)
    processed_train_mask = Path('temp_training/Training/masks/img')
    processed_train_mask.mkdir(parents=True, exist_ok=True)   

    for n,imageName in tqdm(enumerate(train_list)):
        shutil.copy(imageName,processed_train_img/imageName.name)
        os.rename(processed_train_img/imageName.name, Path(str(processed_train_img/imageName.name).replace('/img/', '/img/{}_'.format(n))))

        new_mask = Path(str(processed_train_img/imageName.name).replace('/images/', '/masks/'))
        shutil.copy2(Path(str(imageName).replace('/images/', '/masks/')),new_mask)
        os.rename(new_mask, Path(str(new_mask).replace('/img/', '/img/{}_'.format(n))))


    processed_val_img = Path('temp_training/Validation/images/img')
    processed_val_img.mkdir(parents=True, exist_ok=True)
    processed_val_mask = Path('temp_training/Validation/masks/img')
    processed_val_mask.mkdir(parents=True, exist_ok=True)

    for n,imageName in tqdm(enumerate(valid_list)):
        shutil.copy(imageName,processed_val_img/imageName.name)
        os.rename(processed_val_img/imageName.name, Path(str(processed_val_img/imageName.name).replace('/img/', '/img/{}_'.format(n))))

        new_mask = Path(str(processed_val_img/imageName.name).replace('/images/', '/masks/'))
        shutil.copy(Path(str(imageName).replace('/images/', '/masks/')),Path(str(processed_val_img/imageName.name).replace('/images/', '/masks/')))
        os.rename(new_mask, Path(str(new_mask).replace('/img/', '/img/{}_'.format(n))))

    return int(len(img_list)*(1-valid)), len(img_list)-int(len(img_list)*(1-valid))


def save_model(model, model_id):
    model_json = model.to_json()
    with open("{0}.json".format(model_id),"w") as json_file:
      json_file.write(model_json)
    model.save_weights("{0}.h5".format(model_id))
    print("saved model..! ready to go.")

def load_saved_model(model_id):
    json_file = open('{0}.json'.format(model_id), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("{0}.h5".format(model_id))
    print("Loaded model from disk")
    return loaded_model