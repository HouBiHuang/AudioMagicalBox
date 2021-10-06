import argparse
import librosa
from SpecAugment import spec_augment_tensorflow
import os, sys
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

parser = argparse.ArgumentParser(description='Spec Augment')
parser.add_argument('--audio-path', default='../data/61-70968-0002.wav',
                    help='The audio file.')
parser.add_argument('--time-warp-para', default=80,
                    help='time warp parameter W')
parser.add_argument('--frequency-mask-para', default=27,
                    help='frequency mask parameter F')
parser.add_argument('--time-mask-para', default=100,
                    help='time mask parameter T')
parser.add_argument('--masking-line-number', default=1,
                    help='masking line number')

args = parser.parse_args()
audio_path = "./ㄏㄧㄡ1.wav" #args.audio_path
time_warping_para = args.time_warp_para
time_masking_para = args.frequency_mask_para
frequency_masking_para = args.time_mask_para
masking_line_number = args.masking_line_number

if __name__ == "__main__":

    # Step 0 : load audio file, extract mel spectrogram
    audio, sampling_rate = librosa.load(audio_path,sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio,
                                                     sr=sampling_rate,
                                                     n_mels=256,
                                                     fmax=16000)

    # reshape spectrogram shape to [batch_size, time, frequency, 1]
    shape = mel_spectrogram.shape
    mel_spectrogram = np.reshape(mel_spectrogram, (-1, shape[0], shape[1], 1))

    # Show Raw mel-spectrogram
    #spec_augment_tensorflow.visualization_spectrogram(mel_spectrogram=mel_spectrogram,
    #                                                  title="Raw Mel Spectrogram")

    warped_masked_spectrogram = spec_augment_tensorflow.spec_augment(mel_spectrogram)

    # Show time warped & masked spectrogram
    spec_augment_tensorflow.visualization_tensor_spectrogram(mel_spectrogram=warped_masked_spectrogram,
                                                      title="tensorflow Warped & Masked Mel Spectrogram")
    
    
    #print(warped_masked_spectrogram)
    
    warped_masked_spectrogram = np.reshape(warped_masked_spectrogram, (shape[0], shape[1]))
    print(warped_masked_spectrogram)
    
    #Change spectrogram to audio
    S = librosa.feature.inverse.mel_to_stft(warped_masked_spectrogram)
    y = librosa.griffinlim(S)
    
    #因轉成audio的長度不夠，所以用0填充到8000列
    window = np.zeros(8000)
    y_len = len(y)
    window[0:y_len] = y
    
    wavfile.write("./0Hz.wav", 16000, window)
    #output_signal = librosa.core.spectrum.griffinlim(warped_masked_spectrogram)
    

    
    