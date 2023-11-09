import scipy
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cvlib
from tensorflow.keras.models import Sequential, model_from_json
from scipy import signal
[sr,s]=read("healthy.wav");
fixed_win = 20000;
sliding_win =10000;
windowed_sig=[]

size_of_ip = np.size(s);
c = 0; j = 0;
while (1) :
  New = s[c:c+fixed_win]
  windowed_sig.append(New)
  c = c+sliding_win
  j = j+1
  if(c+fixed_win > size_of_ip):
      break
snippets=np.array(windowed_sig)
final_snip=[]
for i in range (len(snippets[:,1])):
  filt_snip=snippets[i,:];
  snip_norm=filt_snip/np.max(np.abs(filt_snip));
  final_snip.append(snip_norm)
final_snip=np.array(final_snip)

r=len(final_snip[:,1])
nfft=1024
win_length=1024
hop_length=410;sr=4000
audio_rgb_list=[]
audio2d_list=[]
d_shape=224


for i in range (r):
  clip=final_snip[i,:]
  t,f,log_spectrogram = signal.spectrogram(clip, fs=4000, window='hann',nperseg=1024,noverlap=410,nfft=1024)
  log_spectrogram=20*(np.abs(log_spectrogram))
  norm=(log_spectrogram-np.min(log_spectrogram))/(np.max(log_spectrogram)-np.min(log_spectrogram))
  img = norm
  img=cvlib.resize(img, dsize=(d_shape,d_shape), interpolation=cvlib.INTER_CUBIC)
  cmap = plt.get_cmap('hsv')
  rgba_img = cmap(img)
  rgb_img = np.delete(rgba_img, 3, 2)
  clip_rgb=np.flip(rgb_img, 0)
  audio_rgb_list.append(clip_rgb)

print('shape of one spectrogram dataset'+str(np.shape(audio_rgb_list)))

json_file = open('/content/gdrive/MyDrive/ARKA/Asthma_classification/train_on_18_08_22/triplet_mobile_inception.json', 'r')
triplet_model_json = json_file.read()
json_file.close()
triplet_model = model_from_json(triplet_model_json)
# load weights into new model
triplet_model.load_weights("/content/gdrive/MyDrive/ARKA/Asthma_classification/train_on_18_08_22/triplet_2022_08_18_19_34_55.h5")


json_file = open('/content/gdrive/MyDrive/ARKA/Asthma_classification/train_on_18_08_22/classifier.json', 'r')
classifier_model_json = json_file.read()
json_file.close()
classifier_model = model_from_json(classifier_model_json)
# load weights into new model
classifier_model.load_weights("/content/gdrive/MyDrive/ARKA/Asthma_classification/train_on_18_08_22/classifiermodel_1_2022_08_18_19_34_20.h5")

X_test= np.array(audio_rgb_list)
embedding_out = triplet_model.layers[3].predict(X_test)
predicted=classifier_model.predict(embedding_out)
Y_pred=predicted.argmax(axis=-1)

for i in range (len(Y_pred)):
  frame_class_idx=Y_pred[i];
  if frame_class_idx==0:
    print('Frame_'+str(i)+': is predicted as: Asthma')
  else:
    print('Frame_'+str(i)+': is predicted as: Healthy')
