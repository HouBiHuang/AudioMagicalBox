import librosa
import librosa.display
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import numpy as np
import matplotlib.pyplot as plt
import random


def sparse_warp(mel_spectrogram, time_warping_para=2):         
    v, tau = mel_spectrogram.shape[1], mel_spectrogram.shape[2]
        
    horiz_line_thru_ctr = mel_spectrogram[0][v//2] #取得梅爾頻譜圖中心
    
    random_pt = horiz_line_thru_ctr[random.randrange(time_warping_para, tau - time_warping_para)] #沿著水平時間軸隨機取得一點
    w = np.random.uniform((-time_warping_para), time_warping_para) #距離
        
    # Source Points
    src_points = [[[v//2, random_pt[0]]]]
        
    # Destination Points
    dest_points = [[[v//2, random_pt[0] + w]]]

    #How many zero-flow boundary points to include at each image edge. Usage:
    #num_boundary_points=2: 4 corners and one in the middle of each edge (8 points total)
    mel_spectrogram, _ = sparse_image_warp(mel_spectrogram, src_points, dest_points, num_boundary_points=2)
    
    return mel_spectrogram


def time_masking(mel_spectrogram, time_masking_para=3, time_mask_num=1):
    # Step 2 : time masking
    fbank_size = tf.shape(mel_spectrogram)
    n, tau = fbank_size[1], fbank_size[2]

    #產生一堆0&1，再跟mel_spectrogram相乘，乘0的地方代表被遮蔽
    for i in range(time_mask_num):
        #產生遮蔽範圍
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        tau = tf.cast(tau, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # warped_mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, n, tau - t0 - t, 1)),
                          tf.zeros(shape=(1, n, t, 1)),
                          tf.ones(shape=(1, n, t0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def frequency_masking(mel_spectrogram, frequency_masking_para=50, frequency_mask_num=1):
    fbank_size = tf.shape(mel_spectrogram)
    v, n = fbank_size[1], fbank_size[2]

    # Step 3 : frequency masking
    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)
        
        # mel_spectrogram[f0:f0 + f,:] = 0
        mask = tf.concat((tf.ones(shape=(1, v-f0-f, n, 1)),
                          tf.zeros(shape=(1, f, n, 1)),
                          tf.ones(shape=(1, f0, n, 1)),
                          ), 1)
        
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)



def spec_augment(mel_spectrogram):

    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Reshape to [Batch_size, freq, time, 1] for sparse_image_warp func.
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, v, tau, 1))
    
    #時間扭曲
    warped_mel_spectrogram = sparse_warp(mel_spectrogram)

    #時間遮罩
    warped_time_spectrogram = time_masking(warped_mel_spectrogram)

    #頻率遮罩
    warped_time_frequency_sepctrogram = frequency_masking(warped_time_spectrogram)

    #重整shape到原本模樣
    warped_time_frequency_sepctrogram = np.reshape(warped_time_frequency_sepctrogram, (v, tau))
    
    return warped_time_frequency_sepctrogram


def visualization_spectrogram(mel_spectrogram, title):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mel_spectrogram[:, :], ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=16000,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)


def visualization_tensor_spectrogram(mel_spectrogram, title):
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mel_spectrogram[:, :], ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=16000,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)