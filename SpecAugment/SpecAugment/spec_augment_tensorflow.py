import librosa
import librosa.display
import tensorflow as tf
from tensorflow_addons.image import sparse_image_warp
import numpy as np
import matplotlib.pyplot as plt
import random


def sparse_warp(mel_spectrogram, time_warping_para=2):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    
    """
    fbank_size = tf.shape(mel_spectrogram)
    n, v = fbank_size[1], fbank_size[2]

    # Step 1 : Time warping
    # Image warping control point setting.
    # Source
    pt = tf.random.uniform([], time_warping_para, n-time_warping_para, tf.int32) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.cast(src_ctr_pts, dtype=tf.float32)

    # Destination
    w = tf.random.uniform([], -time_warping_para, time_warping_para, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.cast(dest_ctr_pts, dtype=tf.float32)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = sparse_image_warp(mel_spectrogram,
                                        source_control_point_locations,
                                        dest_control_point_locations)
    return warped_image
    """
    
        
    v, tau = mel_spectrogram.shape[1], mel_spectrogram.shape[2]
        
    horiz_line_thru_ctr = mel_spectrogram[0][v//2]
    
    random_pt = horiz_line_thru_ctr[random.randrange(time_warping_para, tau - time_warping_para)] # random point along the horizontal/time axis
    w = np.random.uniform((-time_warping_para), time_warping_para) # distance
        
    # Source Points
    src_points = [[[v//2, random_pt[0]]]]
        
    # Destination Points
    dest_points = [[[v//2, random_pt[0] + w]]]
        
    mel_spectrogram, _ = sparse_image_warp(mel_spectrogram, src_points, dest_points, num_boundary_points=2)
    
    return mel_spectrogram

def time_masking(mel_spectrogram, tau, time_masking_para=3, time_mask_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    # Step 2 : time masking
    fbank_size = tf.shape(mel_spectrogram)
    n, tau = fbank_size[1], fbank_size[2]

    for i in range(time_mask_num):
        t = tf.random.uniform([], minval=0, maxval=time_masking_para, dtype=tf.int32)
        tau = tf.cast(tau, dtype=tf.int32)
        t0 = tf.random.uniform([], minval=0, maxval=tau-t, dtype=tf.int32)

        # warped_mel_spectrogram[f0:f0 + f, :] = 0
        mask = tf.concat((tf.ones(shape=(1, n, tau - t0 - t, 1)),
                          tf.zeros(shape=(1, n, t, 1)),
                          tf.ones(shape=(1, n, t0, 1)),
                          ), 2)
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def frequency_masking(mel_spectrogram, v, frequency_masking_para=50, frequency_mask_num=1):
    """Spec augmentation Calculation Function.

    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.

    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.

    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    fbank_size = tf.shape(mel_spectrogram)
    v, n = fbank_size[1], fbank_size[2]

    # Step 3 : frequency masking
    for i in range(frequency_mask_num):
        f = tf.random.uniform([], minval=0, maxval=frequency_masking_para, dtype=tf.int32)
        f0 = tf.random.uniform([], minval=0, maxval=v-f, dtype=tf.int32)

        # mel_spectrogram[:, t0:t0 + t] = 0
        mask = tf.concat((tf.ones(shape=(1, v-f0-f, n, 1)),
                          tf.zeros(shape=(1, f, n, 1)),
                          tf.ones(shape=(1, f0, n, 1)),
                          ), 1)
        
        mel_spectrogram = mel_spectrogram * mask
    return tf.cast(mel_spectrogram, dtype=tf.float32)


def spec_augment(mel_spectrogram):

    v = mel_spectrogram.shape[0]
    tau = mel_spectrogram.shape[1]

    # Reshape to [Batch_size, time, freq, 1] for sparse_image_warp func.
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, v, tau, 1))
    
    #時間扭曲
    warped_mel_spectrogram = sparse_warp(mel_spectrogram)

    #時間遮罩
    warped_time_spectrogram = time_masking(warped_mel_spectrogram, tau=tau)

    #頻率遮罩
    warped_time_frequency_sepctrogram = frequency_masking(warped_time_spectrogram, v=v)

    #重整shape到原本模樣
    warped_time_frequency_sepctrogram = np.reshape(warped_time_frequency_sepctrogram, (v, tau))
    
    return warped_time_frequency_sepctrogram


def visualization_spectrogram(mel_spectrogram, title):
    """visualizing first one result of SpecAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=16000, x_axis='time')
    #plt.title(title)
    #plt.tight_layout()
    #plt.show()
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mel_spectrogram[:, :], ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=16000,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)


def visualization_tensor_spectrogram(mel_spectrogram, title):
    """visualizing first one result of SpecAugment

    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """

    # Show mel-spectrogram using librosa's specshow.
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :, 0], ref=np.max), y_axis='mel', fmax=16000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    #plt.title(title)
    #plt.tight_layout()
    #plt.show()
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(mel_spectrogram[:, :], ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time',
                             y_axis='mel', sr=16000,
                             fmax=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title=title)
