import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import tensorflow as tf
from PyQt5.QtCore import QTime
import librosa
import scipy.fftpack as sf
#import RPi.GPIO as GPIO

# Parameters
debug_time = 0 #Debug用
debug_acc = 1 #Debug用
led_pin = 8 #LED PIN
word_threshold = 0.5 #預測值>0.5，表示stop
rec_duration = 0.5 #每一段錄音持續時間
#window_stride = 0.5
sample_rate = 48000 #取樣率(依MIC不同而改變)
resample_rate = 8000 #重整後的取樣率(符合MODEL)
num_channels = 1 #音訊深度
num_mfcc = 23 #回傳mfcc的量
model_path = './tflite/recording.tflite'
words = ['ㄏㄧㄡ', 'ㄟ', '吼', '啦', '嗯', '的一個', '的這個', '的那個', '著', '那', '那那個', '阿']#答案對應到的字詞

s = 0 #秒
m = 0 #分
h = 0 #時

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)#取樣音頻數據變數

# GPIO 
#GPIO.setwarnings(False)
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#開始計時
counter = QTime()
counter.restart()

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    #檢查是否降低音頻
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    #檢查是否為整數(只能在整數下執行)
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))
    
    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    #GPIO.output(led_pin, GPIO.LOW)
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    #壓縮成1D張量
    rec = np.squeeze(rec)
    
    # Resample
    #重取樣成8000HZ(以符合訓練模型)
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    #將音訊輸入到window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec
    
    S = np.abs(librosa.stft(window))
    print("np.sum(S):" + str(np.sum(S)))
    #window[:2000] = window[2000:4000]
    #window[2000:4000] = window[4000:6000]
    #window[4000:6000] = window[6000:]
    #window[6000:] = rec
    if np.sum(S) > 1000:
        # Compute features
        mfccs = python_speech_features.base.mfcc(window, #輸入訊號
                                            samplerate=new_fs, #取樣率
                                            winlen=0.256, #音框涵蓋時間
                                            winstep=0.050, #音框間距離
                                            numcep=num_mfcc, #返回係數的量
                                            nfilt=26, #過濾器數量
                                            nfft=2048,#FFT大小
                                            preemph=0.0,#不用預強化濾波器
                                            ceplifter=0,#ROBUST
                                            appendEnergy=False,#係數0的話對被替代成總音框能量的對數
                                            winfunc=np.hanning)#hanning window
        mfccs = mfccs.transpose()
        
        # Make prediction from model
        in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
        #設定輸入張量
        interpreter.set_tensor(input_details[0]['index'], in_tensor)
        #進行預測
        interpreter.invoke()
        #取得輸出張量
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        val = output_data[0]#取得預測值
        val = val.tolist() #np.ndarray to list
        list_val_max = max(val) #取得最大值
        list_val_maxIndex = val.index(max(val)) #取得最大值的索引  
        
        if(list_val_max >= 0.3):#如果預測值>=0.3
            print(words[list_val_maxIndex])#輸出相對應的字詞
            print("MAX:" + str(list_val_max))#輸出預測值當中最大的值
            print(str(h) + "時" + str(m) + "分" + "{:.1f}秒".format(s))
               
    #if debug_acc:
    #    print("pred:" + str(val))#輸出所有預測值
    #    print("MAX:" + str(list_val_max))#輸出預測值當中最大的值
    #    print("MAX_INDEX:" + str(list_val_maxIndex))#輸出最大值的索引值
    
    
# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        s = float(counter.elapsed() / 1000)
        if(s >= 60):
            counter.restart()
            s = 0
            m = m + 1
        if(m == 60):
            m = 0
            h = h + 1
        pass
