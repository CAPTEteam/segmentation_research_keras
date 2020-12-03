
#!/usr/bin/env python3

"""
Created on Tue Jan 30 21:04:26 2019
@author: simadec
Description 
input 
    modelName : nom du modele pour la sauvegarde
    TRAIN_PATH : Chemin du dossier contenant les dossier de datasets pour le format ce référer au nas UMT CAPTE (Dataset segmentation)
    IMG_SIZE : taille des images (le standard est des images de  taille 512*512) SI l'image en entrée est plus grande ou plus petite l'image sera resizer (attention)  
d'autre paramaètres (batch_size early stopping, proportion de la donénes à utiliser pour vlaidation, architecture du modèle ... peuvent être changer)
"""
from utils.prepare_data import 
from utils.utils_inference import *
from keras.layers import Dense, Input
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tqdm
import tensorflow as tf
import xlwt
import segmentation_models as sm
import gc
import numpy as np
import glob
import os
from skimage.io import imread
from PIL import Image
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

threshold=0.5
pathImage='/home/capte-gpu-1/Documents/espaces_personnel/SIMON/simon_segmentation/Dataset_study/Phenomobile_marc/images/'

preprocess_inputEff = sm.get_preprocessing('efficientnetb1')

#modelPath = 'models'

modelName = ['augmentationstudytemp_NoDataAug_epoch-150_dataset-GEVES-mais-veg']
model = load_saved_model(modelName) #, custom_objects={'mean_iou': mean_iou})

print('fINISH TO LOAD THE MODEL!')

for imgN in glob.glob(pathImage+"*.*"):
    print(imgN)
    plt.close('all')
    
    print(imgN)
    #mg = cv2.imread(imgN)
    img = imread(imgN, plugin='matplotlib')[:,:,:3]
    sizeIm = img.shape
    # img = resize(img, (512, 512), mode='constant', preserve_range=True)
    # becarefull, sometimes it is good to resize the images

    if np.max(img)<1.01:        
        img=img*255 # probably has to be removed
        print('warning .. max of the images was less than 1')
        
    sizeIm = img.shape
    sizeImO=sizeIm
    
    sizeIm = img.shape
    if sizeIm[0]<512:
        img = resize(img, (512, np.int(sizeIm[1]*512/sizeIm[0])), mode='constant', preserve_range=True)
        
    if sizeIm[1]<512:
        img = resize(img, (np.int(sizeIm[0]*512/sizeIm[1]), 512), mode='constant', preserve_range=True)
        
    # mettre un if en dessous à faire si seulement c'est nécessaire
    nameIm = imgN[:-4] # Enlebe le .tif .JPG .jpg
    #todo os. path ...

    batch_size = 4
    """ extend image to avoid border problem"""
    bdr = 25
    """ offset to use to extend the image """
    image = np.zeros([sizeIm[0] + bdr * 2, sizeIm[1] + bdr * 2, 3])
    image[bdr:-bdr, bdr:-bdr, :] = img   

    X_test = np.array(list(Patchify(image,512,490)))
    X_test = preprocess_inputEff(X_test)
    preds_test = model.predict(X_test, verbose =1, batch_size = batch_size)

    X_new = unpatchify(preds_test,512,np.shape(image)[:2],490) # probleme ici quand ce n'est pas 512
    X_new = X_new[bdr:-bdr, bdr:-bdr]
    # Reconstruct image
    #X_new = reconstruct_img(preds_test,512,np.shape(img)[:2]) # probleme ici quand ce n'est pas 512
    X_new = resize(X_new, (sizeImO[0], sizeImO[1]), mode='constant', preserve_range=True) 

    # Product image
    X_new = (X_new > threshold).astype(np.uint8)      
    
    X_veg = colorTransform_VegGround(img,X_new,0,1)
    X_ground = colorTransform_VegGround(img,X_new,1,0)
    
    im = Image.fromarray(X_new*255)
    path_visu = os.fspath(os.path.join(os.path.dirname(nameIm), "visualisation_result"))

    """create folder for results"""
    if not os.path.exists(path_visu):
        os.mkdir(path_visu)

    print(os.path.dirname(nameIm))
    im.save(os.path.dirname(nameIm)+ "/visualisation_result/" + os.path.basename(nameIm) + ".png")
    
    print(X_ground.shape)
    if np.max(img)<1.01:
        X_ground=X_ground*255
        X_veg=X_veg*255

    im = Image.fromarray(X_ground.astype('uint8'))
    im.save(os.path.dirname(nameIm)+ "/visualisation_result/" + os.path.basename(nameIm) + os.path.basename(modelName) + '_ground_vglob.jpg')           

    im = Image.fromarray(X_veg.astype('uint8'))
    im.save(os.path.dirname(nameIm)+ "/visualisation_result/" + os.path.basename(nameIm)  + os.path.basename(modelName) + '_veg_vglob.jpg')