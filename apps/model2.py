#Libaries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa as librosa
import librosa as lr
import librosa.display
import IPython.display as ipd
import skimage
import os
import glob
from tqdm import tqdm
import itertools
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import to_categorical
import wave
from scipy.io import wavfile
import pydub
from pydub import AudioSegment
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image

#Important Functions
def extract_relevant(wav_file,t1,t2):
  wav = AudioSegment.from_wav(wav_file)
  wav = wav[1000*t1:1000*t2]
  wav.export("extracted.wav",format='wav')

#Create a melspectrogram
my_cmap = cm.get_cmap('inferno')
def create_melspectrogram(wav_file):
  y,sr = librosa.load(wav_file, duration=3)
  mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
  mels = librosa.feature.melspectrogram(y=y,sr=sr)
  fig = plt.Figure()
  canvas = FigureCanvas(fig)
  p = plt.imshow(librosa.power_to_db(mels,ref=np.max),my_cmap)
  plt.title("Melspectrogram of Audio")
  plt.rcParams["font.family"] = "Sans-serif"
  plt.savefig('melspectrogram.png')
  return mfccs

#main function for predicting uploaded files
def main(file):
    extract_relevant(file, 0,3)
    audio_bytes = file.getvalue()
    x = create_melspectrogram("extracted.wav")
    image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
    st.image("melspectrogram.png",use_column_width=True)
    st.audio(audio_bytes, format='wav',)

    global audio1
    audio = []
    audio.append(x)
    audio = np.asarray(audio)
    audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)

#Main function for predicting sample files
def sample(filename):
    my_cmap = cm.get_cmap('inferno')
    y,sr = librosa.load(filename, duration=3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mels = librosa.feature.melspectrogram(y=y,sr=sr)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    p = plt.imshow(librosa.power_to_db(mels,ref=np.max),my_cmap)
    plt.title("Melspectrogram of Audio")
    plt.rcParams["font.family"] = "Sans-serif"
    plt.savefig('melspectrogram.png')

def app():
    st.markdown(unsafe_allow_html=True, body="<span style='color:black; font-size: 45px'><strong><h4>Arrhythmia Prediction Model 🩺</h4></strong></span>")
    st.sidebar.error('Caution: This is a prediction. Consult a professional for more information!')
    class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
    model_new = tf.keras.models.load_model("heartbeat_disease.h5")
    st.write('Welcome to the Arrhythmia Detector. This app identifies irregular or abnormal heartbeating by analyzing an audio file of heartbeating. The five possible predictions are Artifact (Artifcial noise, not a heartbeat), Extrahls (Extra heartbeating), Extraystole (Premature heartbeating), Murmur (Whoosing or swishing heartbeating), or Normal heartbeating. Refer to `Learn More` for more information!')

    with st.expander('Learn More'):
        st.write('Extrahls, Extrasystole, and Murmur are the main types of heartbeat abnormalities that may be an indicator of Heart Disease. However, if its an isolated incident, then it probably is a sign of heavy excercise, anxiety, drinking, or consumption of medication. However, consult a doctor for more information and if heartbeat abnormailites continue. Remember, this app is simply a diagnostic model!')
        st.write('This model uses a Convolutional Neural Network to process and predict the spectogram (spectrum of frequencies) from heartbeat wav files. It was trained on data from https://www.kaggle.com/kinguistics/heartbeat-sounds, which includes hospital and iphone recordings. Current Accuracy = 79%.')

    st.subheader('Choose an option:')
    user = st.radio("", ("Upload own", "Use sample audio"))

    #If use uploads own audio file, process, predict, and display images
    if user == "Upload own":
        st.subheader('Upload an audio file to get started:')
        uploaded_file = st.file_uploader('', type=["wav"])

        if uploaded_file is None:
            st.subheader("")
        else:
            extract_relevant(uploaded_file, 0,3)
            audio_bytes = uploaded_file.getvalue()
            x = create_melspectrogram("extracted.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)
            st.audio(audio_bytes, format='wav',)

            #global audio1
            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)

            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")

            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")

            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")

            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")

            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")

            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)


            plt.show()
            st.pyplot(fig)

    if user == "Use sample audio":
        st.subheader("Choose a sample audio below:")
        test = st.selectbox("", ("Artifact.wav", "Extrahls.wav", "Extrasystole.wav", "Murmur.wav", "Normal.wav"))
        if (test == "Artifact.wav"):
            artifact = "artifact.wav"
            sample(artifact)

            st.audio("artifact.wav", format='wav')
            x = create_melspectrogram("artifact.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)

            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)
            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")
            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")
            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")
            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")
            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")
            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)
            plt.show()
            st.pyplot(fig)

        if (test == "Extrahls.wav"):
            extrahls = "extrahls.wav"
            sample(extrahls)
            st.audio("extrahls.wav", format='wav')
            x = create_melspectrogram("extrahls.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)

            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)
            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")

            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")

            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")

            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")

            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")

            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)
            #ax.set_title("Prediction Percent of 5 Heartbeat Conditions")

            plt.show()
            st.pyplot(fig)

        if (test == "Extrasystole.wav"):
            extrasystole = "extrasystole.wav"
            sample(extrasystole)
            st.audio("extrasystole.wav", format='wav')
            x = create_melspectrogram("extrasystole.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)

            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)
            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")

            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")

            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")

            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")

            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")

            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)
            #ax.set_title("Prediction Percent of 5 Heartbeat Conditions")

            plt.show()
            st.pyplot(fig)

        if (test == "Murmur.wav"):
            murmur = "murmur.wav"
            sample(murmur)
            st.audio("murmur.wav", format='wav')
            x = create_melspectrogram("murmur.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)

            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)
            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")

            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")

            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")

            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")

            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")

            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)
            #ax.set_title("Prediction Percent of 5 Heartbeat Conditions")

            plt.show()
            st.pyplot(fig)
        if (test == "Normal.wav"):
            normal = "normal1.wav"
            sample(normal)
            st.audio("normal1.wav", format='wav')
            x = create_melspectrogram("normal1.wav")
            image_data = load_img('melspectrogram.png',color_mode='rgba',target_size=(288,432))
            st.image("melspectrogram.png",use_column_width=True)

            audio = []
            audio.append(x)
            audio = np.asarray(audio)
            audio1 = audio.reshape(audio.shape[0], audio.shape[1], audio.shape[2], 1)
            predictions = model_new.predict(audio1)
            prediction = np.argmax(predictions)
            if prediction == 0:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#420a68; font-size: 40px'><strong><h4>Prediction: Artifact</h4></strong></span>")

            if prediction == 1:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#932667; font-size: 50px'><strong><h4>Prediction: Extrahls</h4></strong></span>")

            if prediction == 2:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#dd513a; font-size: 50px'><strong><h4>Prediction: Extrasystole</h4></strong></span>")

            if prediction == 3:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fca50a; font-size: 50px'><strong><h4>Prediction:  Murmur</h4></strong></span>")

            if prediction == 4:
                st.markdown(unsafe_allow_html=True, body="<span style='color:#fcffa4; font-size: 50px'><strong><h4>Prediction: Normal</h4></strong></span>")

            class_labels = ['Artifact', 'Extrahls', 'Extrasystole', 'Murmur', 'Normal']
            color_data = [1,2,3,4,5]
            my_cmap = cm.get_cmap('inferno')
            my_norm = Normalize(vmin=0, vmax=5)
            fig,ax= plt.subplots(figsize=(3.75, 3.5))
            ax.bar(x = class_labels, height=predictions[0,], color=my_cmap(my_norm(color_data)))
            plt.rcParams["font.family"] = "Sans-serif"
            plt.xticks(rotation=45)
            #ax.set_title("Prediction Percent of 5 Heartbeat Conditions")

            plt.show()
            st.pyplot(fig)
