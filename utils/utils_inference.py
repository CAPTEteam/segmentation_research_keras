
import numpy as np

                  
def Patchify(img, size,sizeOverlap):
    sizeIm = img.shape
    for i in range(np.int(size/2),np.int(sizeIm[0]+size/2),sizeOverlap): # 
        for j in range(np.int(size/2),np.int(sizeIm[1]+size/2),sizeOverlap):         
            if i+size/2 > sizeIm[0]:
                   i=sizeIm[0]-size/2
            if j+size/2 > sizeIm[1]:
                j=sizeIm[1]-size/2
            yield img[np.int(i-size/2):np.int(i+size/2),np.int(j-size/2):np.int(j+size/2)]



def unpatchify(predictions,size,shape,sizeOverlap):
    X_new = np.zeros(shape)
    sum_mask = np.zeros(shape)
    c = 0
    offset = np.floor((size-sizeOverlap)/2)
    for i in range(np.int(size/2),np.int(shape[0]+size/2),sizeOverlap): # probleme ici l'iamge doit faire plus que 512
        for j in range(np.int(size/2),np.int(shape[1]+size/2),sizeOverlap):
            Image_a_rajouter = predictions[c] 
            if i+size/2 > shape[0]:
                i=shape[0]-size/2  
            if j+size/2 > shape[1]:
                j=shape[1]-size/2
            X_new[np.int(i-size/2+offset):np.int(i+size/2-offset),np.int(j-size/2 +offset):np.int(j+size/2-offset)] += np.squeeze(Image_a_rajouter[np.int(offset):np.int(size-offset),np.int(offset):np.int(size-offset),:])
            sum_mask[np.int(i-size/2+offset):np.int(i+size/2-offset),np.int(j-size/2+offset):np.int(j+size/2-offset)] += 1
            c += 1 
    return X_new/sum_mask

def colorTransform_VegGround(img,X_true,alpha_vert,alpha_g):
    alpha = alpha_vert
    color = [97,65,38]
    if np.max(img)<1.01: 
        color = [x / 255 for x in color]
    image=np.copy(img)
    for c in range(3):
        image[:, :, c] =np.where(X_true == 0,image[:, :, c] *(1 - alpha) + alpha * color[c] ,image[:, :, c])
    alpha = alpha_g
    color = [34,139,34]
    if np.max(img)<1.01: 
        color = [x / 255 for x in color]
    for c in range(3):
        image[:, :, c] =np.where(X_true == 1,image[:, :, c] *(1 - alpha) + alpha * color[c] ,image[:, :, c])
    return image
