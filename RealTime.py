#即時
import sounddevice as sd
import numpy as np
import scipy.signal
import python_speech_features
import tensorflow as tf
import librosa
#import RPi.GPIO as GPIO

# Parameters
word_threshold = 0.7 #預測值>0.7
rec_duration = 0.25 #每一段錄音持續時間
sample_rate = 16000 #取樣率(依MIC不同而改變)
num_channels = 1 #音訊深度
model_path = './tflite_normalize/recording9_fbank.tflite'
words = ['backgroundNoise', 'ㄏㄧㄡ', 'ㄟ', '他', '你', '其他', '吼', '啦', '嗯', '好', '我', '的', '著', '那', '阿']

# Sliding window
window = np.zeros(8000)#取樣音頻數據變數

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):
    global window
    #GPIO.output(led_pin, GPIO.LOW)
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    #壓縮成1D張量
    rec = np.squeeze(rec)

    #將音訊輸入到window
    window[0:4000] = window[4000:]
    window[4000:] = rec
    S = np.abs(librosa.stft(window)) #將整個window音訊做stft，並轉成絕對值
    
    if np.sum(S) >= 500: #判斷S的總和是否>=500，如果>=500，代表有講話
        
        window = window.astype(np.float)

        window = (window - window.mean()) / (window.max() - window.min())
    
        # Compute features
        features = python_speech_features.base.logfbank(window,
                                                        samplerate=16000,
                                                        winlen=0.025,
                                                        winstep=0.01,
                                                        nfilt=26,
                                                        nfft=512,
                                                        lowfreq=0,
                                                        highfreq=None,
                                                        preemph=0.97)
        
        # Make prediction from model
        in_tensor = np.float32(features.reshape(1, features.shape[0], features.shape[1], 1))
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
        
        if(list_val_max > word_threshold):
            print(words[list_val_maxIndex])#輸出相對應的字詞
            print("MAX:" + str(list_val_max))#輸出預測值當中最大的值
    
# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        
        pass

