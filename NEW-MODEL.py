from keras.models import Model
from keras.layers import *
import keras
from keras.optimizers import *

from keras import backend as K

smooth = 0.0000001

def aspp(x,out_shape):
  b0=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(x)
  b0=BatchNormalization()(b0)
  b0=Activation("relu")(b0)

  #b5=DepthwiseConv2D((3,3),dilation_rate=(3,3),padding="same",use_bias=False)(x)
  #b5=BatchNormalization()(b5)
  #b5=Activation("relu")(b5)
  #b5=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b5)
  #b5=BatchNormalization()(b5)
  #b5=Activation("relu")(b5)
  
  b1=DepthwiseConv2D((3,3),dilation_rate=(3,3),padding="same",use_bias=False)(x)
  b1=BatchNormalization()(b1)
  b1=Activation("relu")(b1)
  b1=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b1)
  b1=BatchNormalization()(b1)
  b1=Activation("relu")(b1)
  
  b2=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
  b2=BatchNormalization()(b2)
  b2=Activation("relu")(b2)
  b2=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b2)
  b2=BatchNormalization()(b2)
  b2=Activation("relu")(b2)	
  
  b3=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
  b3=BatchNormalization()(b3)
  b3=Activation("relu")(b3)
  b3=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b3)
  b3=BatchNormalization()(b3)
  b3=Activation("relu")(b3)
  
  b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
  b4=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b4)
  b4=BatchNormalization()(b4)
  b4=Activation("relu")(b4)
  #b4=UpSampling2D((out_shape,out_shape), interpolation='bilinear')(b4)
  #x=Concatenate()([b4,b0,b1,b2,b3])
  return x
################
def aspp2(x,out_shape):
  b0=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(x)
  b0=BatchNormalization()(b0)
  b0=Activation("relu")(b0)

  #b5=DepthwiseConv2D((3,3),dilation_rate=(3,3),padding="same",use_bias=False)(x)
  #b5=BatchNormalization()(b5)
  #b5=Activation("relu")(b5)
  #b5=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b5)
  #b5=BatchNormalization()(b5)
  #b5=Activation("relu")(b5)

  b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
  b1=BatchNormalization()(b1)
  b1=Activation("relu")(b1)
  b1=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b1)
  b1=BatchNormalization()(b1)
  b1=Activation("relu")(b1)

  b2=DepthwiseConv2D((3,3),dilation_rate=(12,12),padding="same",use_bias=False)(x)
  b2=BatchNormalization()(b2)
  b2=Activation("relu")(b2)
  b2=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b2)
  b2=BatchNormalization()(b2)
  b2=Activation("relu")(b2)

  b3=DepthwiseConv2D((3,3),dilation_rate=(18,18),padding="same",use_bias=False)(x)
  b3=BatchNormalization()(b3)
  b3=Activation("relu")(b3)
  b3=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b3)
  b3=BatchNormalization()(b3)
  b3=Activation("relu")(b3)

  b4=AveragePooling2D(pool_size=(out_shape,out_shape))(x)
  b4=SeparableConv2D(256,(1,1),padding="same",use_bias=False)(b4)
  b4=BatchNormalization()(b4)
  b4=Activation("relu")(b4)
  #b4=UpSampling2D((out_shape,out_shape), interpolation='bilinear')(b4)
  #x=Concatenate()([b4,b0,b1,b2,b3])
  return x


#################
def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

def bn_relu(input_tensor):
    """It adds a Batch_normalization layer before a Relu
    """
    input_tensor = BatchNormalization(axis=3)(input_tensor)
    return Activation("relu")(input_tensor)


def contr_arm(input_tensor, filters, kernel_size):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
       TO DO: remove keras.layers.add and replace it with add only
    """

    x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x = SeparableConv2D(filters, kernel_size, padding='same')(x)
    x = bn_relu(x)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

    x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x1 = bn_relu(x1)

    x1 = concatenate([input_tensor, x1], axis=3)
    x = keras.layers.add([x, x1])
    x = Activation("relu")(x)
    return x


def imprv_contr_arm(input_tensor, filters, kernel_size ):
    """It adds a feedforward signal to the output of two following conv layers in contracting path
    """

    x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x0 = SeparableConv2D(filters, kernel_size, padding='same')(x)
    x0 = bn_relu(x0)

    x = SeparableConv2D(filters, kernel_size, padding='same')(x0)
    x = bn_relu(x)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

    x1 = SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x1 = bn_relu(x1)

    x1 = concatenate([input_tensor, x1], axis=3)

    x2 = SeparableConv2D(filters, kernel_size_b, padding='same')(x0)
    x2 = bn_relu(x2)

    x = keras.layers.add([x, x1, x2])
    x = Activation("relu")(x)
    return x


def bridge(input_tensor, filters, kernel_size):
    """It is exactly like the identity_block plus a dropout layer. This block only uses in the valley of the UNet
    """

    x = SeparableConv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x = SeparableConv2D(filters, kernel_size, padding='same')(x)
    x = Dropout(.15)(x)
    x = bn_relu(x)

    filters_b = filters // 2
    kernel_size_b = (kernel_size[0]-2, kernel_size[0]-2)  # creates a kernl size of (1,1) out of (3,3)

    x1 =SeparableConv2D(filters_b, kernel_size_b, padding='same')(input_tensor)
    x1 = bn_relu(x1)

    x1 = concatenate([input_tensor, x1], axis=3)
    x = keras.layers.add([x, x1])
    x = Activation("relu")(x)
    return x


def conv_block_exp_path(input_tensor, filters, kernel_size):
    """It Is only the convolution part inside each expanding path's block
    """

    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = bn_relu(x)
    return x


def conv_block_exp_path3(input_tensor, filters, kernel_size):
    """It Is only the convolution part inside each expanding path's block
    """

    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = bn_relu(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = bn_relu(x)

    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = bn_relu(x)
    return x


def add_block_exp_path(input_tensor1, input_tensor2, input_tensor3):
    """It is for adding two feed forwards to the output of the two following conv layers in expanding path
    """

    x = keras.layers.add([input_tensor1, input_tensor2, input_tensor3])
    x = Activation("relu")(x)
    return x


def improve_ff_block4(input_tensor1, input_tensor2 ,input_tensor3, input_tensor4, pure_ff):
    """It improves the skip connection by using previous layers feature maps
       TO DO: shrink all of ff blocks in one function/class
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = concatenate([x1, input_tensor1], axis=3)
        x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = concatenate([x2, input_tensor2], axis=3)
    x2 = MaxPooling2D(pool_size=(4, 4))(x2)

    for ix in range(7):
        if ix == 0:
            x3 = input_tensor3
        x3 = concatenate([x3, input_tensor3], axis=3)
    x3 = MaxPooling2D(pool_size=(8, 8))(x3)

    for ix in range(15):
        if ix == 0:
            x4 = input_tensor4
        x4 = concatenate([x4, input_tensor4], axis=3)
    x4 = MaxPooling2D(pool_size=(16, 16))(x4)

    x = keras.layers.add([x1, x2, x3, x4, pure_ff])
    x = Activation("relu")(x)
    return x


def improve_ff_block3(input_tensor1, input_tensor2, input_tensor3, pure_ff):
    """It improves the skip connection by using previous layers feature maps
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = concatenate([x1, input_tensor1], axis=3)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = concatenate([x2, input_tensor2], axis=3)
    x2 = MaxPooling2D(pool_size=(4, 4))(x2)

    for ix in range(7):
        if ix == 0:
            x3 = input_tensor3
        x3 = concatenate([x3, input_tensor3], axis=3)
    x3 = MaxPooling2D(pool_size=(8, 8))(x3)

    x = keras.layers.add([x1, x2, x3, pure_ff])
    x = Activation("relu")(x)
    return x


def improve_ff_block2(input_tensor1, input_tensor2, pure_ff):
    """It improves the skip connection by using previous layers feature maps
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = concatenate([x1, input_tensor1], axis=3)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    for ix in range(3):
        if ix == 0:
            x2 = input_tensor2
        x2 = concatenate([x2, input_tensor2], axis=3)
    x2 = MaxPooling2D(pool_size=(4, 4))(x2)

    x = keras.layers.add([x1, x2, pure_ff])
    x = Activation("relu")(x)
    return x


def improve_ff_block1(input_tensor1, pure_ff):
    """It improves the skip connection by using previous layers feature maps
    """

    for ix in range(1):
        if ix == 0:
            x1 = input_tensor1
        x1 = concatenate([x1, input_tensor1], axis=3)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)

    x = keras.layers.add([x1, pure_ff])
    x = Activation("relu")(x)
    return x


def model_arch(input_rows=256, input_cols=256, num_of_channels=3, num_of_classes=1):
    inputs = Input((input_rows, input_cols, num_of_channels))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    conv1 = contr_arm(conv1, 32, (3, 3))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = contr_arm(pool1, 64, (3, 3))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = contr_arm(pool2, 128, (3, 3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = contr_arm(pool3, 256, (3, 3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = imprv_contr_arm(pool4, 512, (3, 3))
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = bridge(pool5, 1024, (3, 3))
    
    conv6  = aspp2(conv6,input_rows/32)

    convT7 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv6)
    prevup7 = improve_ff_block4(input_tensor1=conv4, input_tensor2=conv3, input_tensor3=conv2, input_tensor4=conv1, pure_ff=conv5)
    up7 = concatenate([convT7, prevup7], axis=3)
    conv7 = conv_block_exp_path3(input_tensor=up7, filters=512, kernel_size=(3, 3))
    conv7 = add_block_exp_path(conv7, conv5, convT7)

    convT8 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv7)
    prevup8 = improve_ff_block3(input_tensor1=conv3, input_tensor2=conv2, input_tensor3=conv1, pure_ff=conv4)
    up8 = concatenate([convT8, prevup8], axis=3)
    conv8 = conv_block_exp_path(input_tensor=up8, filters=256, kernel_size=(3, 3))
    conv8 = add_block_exp_path(input_tensor1=conv8, input_tensor2=conv4, input_tensor3=convT8)

    convT9 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv8)
    prevup9 = improve_ff_block2(input_tensor1=conv2, input_tensor2=conv1, pure_ff=conv3)
    up9 = concatenate([convT9, prevup9], axis=3)
    conv9 = conv_block_exp_path(input_tensor=up9, filters=128, kernel_size=(3, 3))
    conv9 = add_block_exp_path(input_tensor1=conv9, input_tensor2=conv3, input_tensor3=convT9)

    convT10 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv9)
    prevup10 = improve_ff_block1(input_tensor1=conv1, pure_ff=conv2)
    up10 = concatenate([convT10, prevup10], axis=3)
    conv10 = conv_block_exp_path(input_tensor=up10, filters=64, kernel_size=(3, 3))
    conv10 = add_block_exp_path(input_tensor1=conv10, input_tensor2=conv2, input_tensor3=convT10)

    convT11 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv10)
    up11 = concatenate([convT11, conv1], axis=3)
    conv11 = conv_block_exp_path(input_tensor=up11, filters=32, kernel_size=(3, 3))
    conv11 = add_block_exp_path(input_tensor1=conv11, input_tensor2=conv1, input_tensor3=convT11)

    conv12 = Conv2D(num_of_classes, (1, 1), activation='sigmoid')(conv11)

    return Model(inputs=[inputs], outputs=[conv12])



model = model_arch(input_rows=384, input_cols=384, num_of_channels=3, num_of_classes=1)
model.compile(optimizer = Adam(lr = 1e-4), loss = jacc_coef, metrics = [jacc_coef,'accuracy'])
len(model.layers)
##################################################
#####################################################
import os
import tensorflow as tf
from tensorflow import keras
import random
from keras.callbacks import TensorBoard
#from keras import backend as K
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np
from PIL import Image
from cxn_model import *
 
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
 
TRAIN_PATH_R = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B4/train/'  #change path 
TRAIN_PATH_G = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B3/train/'
TRAIN_PATH_B = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B2/train/'
 
TEST_PATH_R = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B4/test/'
TEST_PATH_G = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B3/test/'
TEST_PATH_B = '/root/Downloads/dataset-20210724T151100Z-001/dataset/B2/test/'
 
X_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((350, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
 
tr=np.zeros(350)
te=np.zeros(10)
 
for i in range(350):
       tr[i]=i;
for i in range(10):
       te[i]=i;
 
 
for n, id_ in tqdm(enumerate(tr),total=350):
       red    = Image.open(TRAIN_PATH_R + str(int(id_)) + '.png').convert('L')
       green  = Image.open(TRAIN_PATH_G + str(int(id_)) + '.png').convert('L')
       blue   = Image.open(TRAIN_PATH_B + str(int(id_)) + '.png').convert('L')
       
       rgb = Image.merge("RGB",(red,green,blue))
       img_b = np.asarray(rgb) 
       
       #img_r = imread(TRAIN_PATH_R + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_g = imread(TRAIN_PATH_G + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       #img_b = imread(TRAIN_PATH_B + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
 
       #img_r = resize(img_r, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_g = resize(img_g, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
       img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
       X_train[n] = img_b
       
       #mask = Image.open('/content/drive/My Drive/Colab Notebooks/dataset/BQA/train/' + str(int(102+id_)) + '.png').convert('L')
       mask = imread('/root/Downloads/dataset-20210724T151100Z-001/dataset/BQA/train/' + str(int(id_)) + '.png')[:,:,:IMG_CHANNELS]
       mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       Y_train[n] =mask1/255.0
       for io in range(256):
         for jo in range(256):
           if (Y_train[n][io][jo]>0.3):
             Y_train[n][io][jo]=1
           else:
             Y_train[n][io][jo]=0
       
 
# for test images 
 
X_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_test = np.zeros((10, IMG_WIDTH, IMG_WIDTH, 1), dtype=np.float32)
img = np.zeros((IMG_WIDTH, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
 
 
for n, id_ in tqdm(enumerate(te),total=10):
       red    = Image.open(TEST_PATH_R + str(170+int(id_)) + '.png').convert('L')
       green  = Image.open(TEST_PATH_G + str(170+int(id_)) + '.png').convert('L')
       blue   = Image.open(TEST_PATH_B + str(170+int(id_)) + '.png').convert('L')
       
       rgb = Image.merge("RGB",(red,green,blue))
       img_b = np.asarray(rgb)  
 
       img_b = resize(img_b, (IMG_HEIGHT, IMG_WIDTH, 3), mode='constant', preserve_range=True)
       img_b=img_b/255.0
       #for i in range(256):
       #       img[i] = np.concatenate((img_r[i],img_g[i],img_b[i]), axis=1)
       
       X_test[n] = img_b
 
       mask = imread('/root/Downloads/dataset-20210724T151100Z-001/dataset/BQA/test/' + str(int(170+id_)) + '.png')[:,:,:IMG_CHANNELS]
       mask1 = resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
       #mask1 = np.asarray(mask)
       #mask1 = resize(mask1, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
 
       Y_test[n] =mask1/255.0
       for io in range(256):
         for jo in range(256):
           if (Y_test[n][io][jo]>0.3):
             Y_test[n][io][jo]=1
           else:
             Y_test[n][io][jo]=0


model = model_arch(input_rows=256, input_cols=256, num_of_channels=3, num_of_classes=1)
model.compile(optimizer = Adam(lr = 1e-4), loss = jacc_coef, metrics = [jacc_coef,'accuracy'])


#checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5' , verbose=1, save_best_only=True)
#callbacks = [tf.keras.callbacks.EarlyStopping(patience=50, monitor='val_loss'),tf.keras.callbacks.TensorBoard(log_dir="logs")]


results = model.fit(X_train, Y_train, validation_split=0.05, batch_size=2, epochs=70, verbose=1)   #, callbacks=[cp_callback])


preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.95):], verbose=1)
preds_test = model.predict(X_test, verbose=1)


preds_train_t = (preds_train > 0.5).astype(np.float32)
preds_val_t = (preds_val > 0.5).astype(np.float32)
preds_test_t = (preds_test > 0.5).astype(np.float32)

train_acc = model.evaluate(X_train, Y_train, verbose=1)
test_acc = model.evaluate(X_test, Y_test, verbose=1)
##########################################################
###############################################################
from sklearn.metrics import confusion_matrix

def precision(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask).ravel()
  prec = tp/(tp+fp)
  return(prec)

####recall---
def recall(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask).ravel()
  rec = tp/(tp+fn)
  return(rec)

###f1 score--

def f1_score(prec,rec):
  f1 = 2*(prec*rec)/(prec+rec)
  return f1

  ### jaccard
def jaccard(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask).ravel()
  rec = tp/(tp+fn+fp)
  return(rec)

  ### jaccard
def Overall(gt,mask):
  gt = gt.flatten()
  mask = mask.flatten()
  tn,fp,fn,tp = confusion_matrix(gt,mask).ravel()
  rec = (tp+tn)/(tp+fp+fn+tn)
  return(rec)





###aji score

def get_fast_aji(true, pred):

    true = np.copy(true) # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros([len(true_id_list) -1,
                               len(pred_id_list) -1], dtype=np.float64)
    pairwise_union = np.zeros([len(true_id_list) -1,
                               len(pred_id_list) -1], dtype=np.float64)

    # caching pairwise
    for true_id in true_id_list[1:]: # 0-th is background
        t_mask = true_masks[int(true_id)]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0: # ignore
                continue # overlaping background
            p_mask = pred_masks[int(pred_id)]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[int(true_id)-1, int(pred_id)-1] = inter
            pairwise_union[int(true_id)-1, int(pred_id)-1] = total - inter
    #
    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    #
    paired_true = (list(paired_true + 1)) # index to instance ID
    paired_pred = (list(paired_pred + 1))
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array([idx for idx in true_id_list[1:] if idx not in paired_true])
    unpaired_pred = np.array([idx for idx in pred_id_list[1:] if idx not in paired_pred])
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()
    #
    aji_score = overall_inter / overall_union
    return aji_score

sum = 0
for i in range(len(Y_test)):
  sum = sum + precision(Y_test[i],preds_test_t[i])
prec = sum/len(Y_test)

sum = 0
for i in range(len(Y_test)):
  sum = sum + recall(Y_test[i],preds_test_t[i])
rec = sum/len(Y_test)

sum = 0
for i in range(len(Y_test)):
  sum = sum + jaccard(Y_test[i],preds_test_t[i])
jaccard1 = sum/len(Y_test)


sum = 0
for i in range(len(Y_test)):
  sum = sum + Overall(Y_test[i],preds_test_t[i])
Overall1 = sum/len(Y_test)


f1 = f1_score(prec,rec)
aji = get_fast_aji(Y_test,preds_test_t)

print("Jaccard Index", jaccard1)
print("final f1", f1)
print("final precision",prec)
print("final recall",rec)
print("Overall Accuracy",Overall1)
print("final aji",aji)
