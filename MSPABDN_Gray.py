# import packages

from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,Concatenate,Conv2DTranspose,Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
tf_device='/gpu:7'
# create CNN model
input_img=Input(shape=(None,None,1))
x=Activation('relu')(input_img)
x1=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x1)
x=Conv2D(32,(3,3),dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=4,padding="same")(x)

x=Activation('relu')(x)
x11=Conv2D(32,(3,3),dilation_rate=5,padding="same")(x)

# MSPA
# conv1
x111 = Conv2D(32,(1,1),padding="same")(x11)

x1111 = Conv2D(32,(1,1),padding="same")(x111)
x=Activation('relu')(x1111)
x=Conv2D(32,(1,1),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(1,1),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(1,1),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(1,1),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(1,1),dilation_rate=1,padding="same")(x)

x=Conv2D(32,(1,1),padding="same")(x)

x=Activation('sigmoid')(x)
x_joint1 = Multiply()([x,x111])

### conv2

x111 = Conv2D(32,(2,2),padding="same")(x11)

x1111 = Conv2D(32,(2,2),padding="same")(x111)
x=Activation('relu')(x1111)
x=Conv2D(32,(2,2),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(2,2),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(2,2),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(2,2),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(2,2),dilation_rate=1,padding="same")(x)

x=Conv2D(32,(2,2),padding="same")(x)

x=Activation('sigmoid')(x)
x_joint2 = Multiply()([x,x111])

# conv3

x111 = Conv2D(32,(3,3),padding="same")(x11)

x1111 = Conv2D(32,(3,3),padding="same")(x111)
x=Activation('relu')(x1111)
x=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x=Conv2D(32,(3,3),padding="same")(x)

x=Activation('sigmoid')(x)
x_joint3 = Multiply()([x,x111])

# conv4

x111 = Conv2D(32,(4,4),padding="same")(x11)

x1111 = Conv2D(32,(4,4),padding="same")(x111)
x=Activation('relu')(x1111)
x=Conv2D(32,(4,4),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(4,4),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(4,4),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(4,4),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(4,4),dilation_rate=1,padding="same")(x)

x=Conv2D(32,(4,4),padding="same")(x)

x=Activation('sigmoid')(x)
x_joint4 = Multiply()([x,x111])

# conv 5

x111 = Conv2D(32,(5,5),padding="same")(x11)

x1111 = Conv2D(32,(5,5),padding="same")(x111)
x=Activation('relu')(x1111)
x=Conv2D(32,(5,5),dilation_rate=1,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(5,5),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(5,5),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(5,5),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(5,5),dilation_rate=1,padding="same")(x)

x=Conv2D(32,(5,5),padding="same")(x)

x=Activation('sigmoid')(x)
x_joint5 = Multiply()([x,x111])

x_com1 = Add()([x_joint1, x_joint2])
x_com2 = Add()([x_com1, x_joint3])
x_com3 = Add()([x_com2, x_joint4])
x_joint6 = Add()([x_com3, x_joint5])

x=Activation('relu')(x_joint6)
x=Conv2D(32,(3,3),dilation_rate=5,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=4,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(32,(3,3),dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x2=Conv2D(32,(3,3),dilation_rate=1,padding="same")(x)

x3 = Subtract()([x2, x1])

x4=Conv2D(1,(3,3),padding="same")(x3)
x5 = Add()([x4, input_img])
model = Model(inputs=input_img, outputs=x5)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest")
def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(0,55)
        trueNoiseBatch=np.random.normal(0,noise/255.0,batch.shape)
        noisyImagesBatch=batch+trueNoiseBatch
        yield (noisyImagesBatch,trueNoiseBatch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(lr=0.001)
def custom_loss(y_true,y_pred):
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=config.epochs,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('./Pretrained_models/MSPABDN_Gray.h5')
