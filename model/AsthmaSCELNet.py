import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
!pip install mat73
import mat73
!pip install keras==2.2.4
!pip install tensorflow==2.8.1
sig_dict=mat73.loadmat('/content/gdrive/MyDrive/ARKA/Asthma_classification/all_sigs.mat')
sig=sig_dict['all_sig'];
sig=sig.transpose()
labels=pd.read_excel('/content/gdrive/MyDrive/ARKA/Asthma_classification/labels.xlsx',header=None)
l=np.array(labels)
l_new=np.reshape(l,(l.shape[0],))
import librosa
r=len(sig[:,1])
nfft=1024
win_length=1024
hop_length=410;sr=4000
audio_rgb_list=[]
d_shape=224

import cv2 as cvlib
for i in range (r):
  clip=sig[i,:]
  mel_spec=librosa.feature.melspectrogram( y=clip, sr=4000,n_mels=64, n_fft=1024, hop_length=410, win_length=1024, window='hann')
  log_spectrogram = librosa.amplitude_to_db(mel_spec)
  norm=(log_spectrogram-np.min(log_spectrogram))/(np.max(log_spectrogram)-np.min(log_spectrogram))
  img = norm
  img=cvlib.resize(img, dsize=(d_shape,d_shape), interpolation=cvlib.INTER_CUBIC)
  cmap = plt.get_cmap('jet')
  rgba_img = cmap(img)
  rgb_img = np.delete(rgba_img, 3, 2)
  clip_rgb=np.flip(rgb_img, 0)
  audio_rgb_list.append(clip_rgb)

X=np.array(audio_rgb_list)
from sklearn.model_selection import train_test_split
X_train_1,X_test,Y_train_1,Y_test=train_test_split(X,l_new,test_size=0.1,random_state=10)
X_train,X_val,Y_train,Y_val=train_test_split(X_train_1,Y_train_1,test_size=0.1,random_state=22)

import random
def plot_triplets(examples):
    plt.figure(figsize=(18, 3))
    for i in range(3):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(np.reshape(examples[i], (224,224,3)))
        plt.xticks([])
        plt.yticks([])
        if i==0:
          plt.title('Anchor');
        elif i==1:
          plt.title('Positive')
        elif i==2:
          plt.title('Negetive')
    plt.show()
def create_batch_train(batch_size):
    x_anchors_train = [];
    x_positives_train = [];
    x_negatives_train = [];

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, X_train.shape[0] - 1)
        x_anchor = X_train[random_index]
        y = Y_train[random_index]

        indices_for_pos = np.squeeze(np.where(Y_train == y)); final_pos_idx=indices_for_pos[random.randint(0, len(indices_for_pos) - 1)];
        indices_for_neg = np.squeeze(np.where(Y_train != y)); final_neg_idx=indices_for_neg[random.randint(0, len(indices_for_neg) - 1)];

        x_positive = X_train[final_pos_idx];
        x_negative = X_train[final_neg_idx];

        x_anchors_train.append(x_anchor) ;
        x_positives_train.append(x_positive)
        x_negatives_train.append(x_negative)

    x_anchors_train=np.array(x_anchors_train)
    x_positives_train=np.array(x_positives_train)
    x_negatives_train=np.array(x_negatives_train)
    return [x_anchors_train, x_positives_train, x_negatives_train]
import tensorflow as tf
from keras.layers import *
from keras.models import Model, Sequential
from keras.layers import Input, Lambda
from keras.regularizers import l2
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
import keras

def mobile_inception(dim):
    print("\nTRAINING ON AsthmaSCELNet:-")


    def block(x, filters, reps):
        for _ in range(reps):

            t1 = Conv2D(filters[0], kernel_size = (1,1))(x)
            t1 = LeakyReLU()(t1)

            t2 = DepthwiseConv2D(kernel_size = (3,3), strides = 1, padding = 'same')(x)
            t2 = LeakyReLU()(t2)
            t2 = Conv2D(filters[1], kernel_size = (1,1))(t2)
            t2 = LeakyReLU()(t2)

            t3 = DepthwiseConv2D(kernel_size = (5,5), strides = 1, padding = 'same')(x)
            t3 = LeakyReLU()(t3)
            t3 = Conv2D(filters[2], kernel_size = (1,1))(t3)
            t3 = LeakyReLU()(t3)

            t4 = MaxPool2D(pool_size = (3,3), strides = 1, padding = 'same')(x)
            t4 = Conv2D(filters[3], kernel_size = (1,1))(t4)
            t4 = LeakyReLU()(t4)

            x_cat = Concatenate()([t1, t2, t3, t4])
            x_out = tf.keras.layers.Add()([x_cat, x])
        return x_out


    input = Input(shape = dim)

    k = 16

    x = Conv2D(filters = k, kernel_size = (3,3), strides = 2, padding = 'same')(input)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')(x)

    x = DepthwiseConv2D(kernel_size = (3,3), strides = 1, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters = 2*k, kernel_size = (1,1))(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)

    x = block(x, [k, k, k, k], reps = 2)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)

    x = GlobalAveragePooling2D()(x)
    output=  Dense(40, 'relu')(x)

    model = Model(inputs = input, outputs = output)

    return model
alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)
dim=(224,224,3)
input_anchor = Input(shape=(dim))
input_positive = tf.keras.layers.Input(shape=(dim))
input_negative = tf.keras.layers.Input(shape=(dim))

embedding_anchor = base_model(input_anchor)
embedding_positive = base_model(input_positive)
embedding_negative = base_model(input_negative)

output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

net = Model([input_anchor, input_positive, input_negative], output)
net.summary()

emb_size=40
def train_data_generator(batch_size=256):
    while True:
        x_t = create_batch_train(batch_size)
        y_t = np.zeros((batch_size, 3*emb_size))
        yield x_t, y_t
steps_per_epoch = int(X_train.shape[0]/batch_size)
opt =tf.keras.optimizers.Adam(learning_rate=0.008)
net.compile(loss=triplet_loss, optimizer=opt)

batch_size = 64
epochs = 400
history =net.fit(train_data_generator(batch_size),steps_per_epoch=steps_per_epoch,epochs=epochs, verbose=1) 

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('Training Loss',size = 20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import pandas as pd


def scatter(x, labels):
    # Create a scatter plot of all the
    # the embeddings of the model.
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
    # We create a scatter plot.
    plt.figure(figsize=(10,8))
    sns.scatterplot(x[:,0], x[:,1], hue=labels)
# Using the newly trained model compute the embeddings
# for a number images
from sklearn.manifold import TSNE
X_train_trm = net.layers[3].predict(X_train)

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
train_tsne_embeds = tsne.fit_transform(X_train_trm)
scatter(train_tsne_embeds, Y_train)
plt.title('Training instances embedding')
X_val_trm = net.layers[3].predict(X_val)

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
val_tsne_embeds = tsne.fit_transform(X_val_trm)
scatter(val_tsne_embeds, Y_val)
plt.title('Validation instances embedding')
X_test_trm = net.layers[3].predict(X_test)

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
test_tsne_embeds = tsne.fit_transform(X_test_trm)
scatter(test_tsne_embeds, Y_test)
plt.title('Testing instances embedding')

import os
from datetime import date
today = date.today()
f_date=today.strftime("%d_%m_%y")
os.mkdir("/content/gdrive/My Drive/ARKA/Asthma_classification/train_on_"+(f_date))
new_loc="/content/gdrive/My Drive/ARKA/Asthma_classification/train_on_"+(f_date)
from keras.models import model_from_json
model_json = net.to_json()
with open(new_loc+"/triplet_mobile_inception.json", "w") as json_file:
    json_file.write(model_json)
from pytz import timezone
from datetime import datetime
ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y_%m_%d_%H_%M_%S')
# save model
net.save_weights(new_loc+'/triplet'+'_'+str(ind_time)+'.h5')
print("===================================done=============================================================")
output_neurons=2
output_activation='sigmoid'

ips = Input(shape=(40,))
#x = net.layers[3] (ips)
t1= Dense(20, 'relu')(ips)
t1=Dropout(0.2)(t1)
t1= Dense(20, 'relu')(t1)
t1=Dropout(0.2)(t1)
output = Dense(output_neurons, output_activation)(t1)

classifier_model = Model(inputs=ips, outputs=output)
classifier_model.summary()
X_train_trm = net.layers[3].predict(X_train)
Y_train_df=pd.DataFrame(Y_train,columns = ['Lung_Sound'])
class_label_onehot_ytr=pd.get_dummies(Y_train_df)
Y_train_l=np.array(class_label_onehot_ytr);

print(Y_train_l.shape)
X_val_trm = net.layers[3].predict(X_val)
Y_val_df=pd.DataFrame(Y_val,columns = ['Lung_Sound'])
class_label_onehot_yvl=pd.get_dummies(Y_val_df)
Y_val_l=np.array(class_label_onehot_yvl);

print(Y_val_l.shape)
X_test_trm = net.layers[3].predict(X_test)
Y_test_df=pd.DataFrame(Y_test,columns = ['Lung_Sound'])
class_label_onehot_ytst=pd.get_dummies(Y_test_df)
Y_tst_l=np.array(class_label_onehot_ytst);

print(Y_tst_l.shape)
opt =tf.keras.optimizers.Adam(learning_rate=0.008)
classifier_model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history = classifier_model.fit(X_train_trm, Y_train_l, batch_size=128, epochs=400, verbose=1,validation_data=(X_val_trm, Y_val_l))

from keras.models import model_from_json
model_json_classifier = classifier_model.to_json()
with open(new_loc+"/classifier.json", "w") as json_file:
    json_file.write(model_json_classifier)
from pytz import timezone
from datetime import datetime
ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y_%m_%d_%H_%M_%S')
# save model
classifier_model.save_weights(new_loc+'/classifiermodel_'+str(1)+'_'+str(ind_time)+'.h5')
print("===================================done=============================================================")

results=classifier_model.evaluate(X_test_trm,Y_tst_l,batch_size=32,verbose=1)
print('Test loss:', results[0])
print('Test accuracy:', results[1])

from tqdm import tqdm
# prediction using CNN
predicted=classifier_model.predict(X_test_trm,batch_size=32,verbose=0)
Y_pred=predicted.argmax(axis=-1)
Y_predicted=pd.DataFrame(Y_tst_l,columns=['Asthma','Healthy'])
Y_ori=[];Asthma_t=0;Normal_t=0;
for index,row in tqdm(Y_predicted.iterrows()):
    if row['Asthma']==1:
        Asthma_t=Asthma_t+1
        Y_ori.append(0)
    elif row['Healthy']==1:
        Normal_t=Normal_t+1
        Y_ori.append(1)

from sklearn.metrics import confusion_matrix
import sklearn
cm=confusion_matrix(Y_ori,Y_pred)
print('Confusion Matrix');
print(cm)
precision = sklearn.metrics.precision_score(Y_ori,Y_pred)
print('precision==   '+str(precision))
accuracy=np.diag(cm).sum()/cm.sum().sum()
print('Accuracy==    '+str(accuracy))
recall = sklearn.metrics.recall_score(Y_ori,Y_pred)
print('Recall==      '+str(recall))
F1 = sklearn.metrics.f1_score(Y_ori,Y_pred)
print('F1-Score==    '+str(F1))
K_cappa = sklearn.metrics.cohen_kappa_score(Y_ori,Y_pred)
print('Kcappa==      '+str(K_cappa))

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_ori,Y_pred)
cm_norm=confusion_matrix(Y_ori,Y_pred,normalize='true')
cm_df = pd.DataFrame(cm,
                     index = ['Asthma','Healthy'],
                     columns = ['Asthma','Healthy'])
plt.figure(figsize=(10,4))
plt.subplot(121)
sns.heatmap(cm_df,annot=True,cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

cm_df_norm = pd.DataFrame(cm_norm,
                     index = ['Asthma','Healthy'],
                     columns = ['Asthma','Healthy'])

plt.subplot(122)
sns.heatmap(cm_df_norm,annot=True,cmap="Blues")
plt.title('Normalised Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

