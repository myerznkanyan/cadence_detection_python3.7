import argparse
import matplotlib.pyplot as plt
from unittest import TestCase
import scipy.io.wavfile as wav
from scipy.fftpack import fft, rfft, fftshift
from scipy.signal import butter, sosfilt, sosfreqz, lfilter
#from scipy.fft import fftshift
import copy
from sklearn.preprocessing import normalize
from numba.decorators import jit as optional_jit
from cv2 import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage
from scipy import signal
import numpy as np
import sys
import json
from sklearn.preprocessing import normalize
import logging
import wave
import os
import librosa
import librosa.display

log = logging.getLogger(__name__)
ch = logging.StreamHandler()
ch.setFormatter = logging.Formatter('%(message)s')
log.addHandler(ch)
log.setLevel(logging.INFO)







FREQ = 44100
rate = 44100
sr = 44100

# Most of the Spectrograms and Inversion are taken from: https://gist.github.com/kastnerkyle/179d6e9a88202ab0a2fe


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

fft_size = 2048  # window size for the FFT
step_size = fft_size // 16  # distance to slide along the window (in time)
spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
lowcut = 500  # Hz # Low cut for our butter bandpass filter
highcut = 15000  # Hz # High cut for our butter bandpass filter
# For mels
n_mel_freq_components = 64  # number of mel frequency channels
shorten_factor = 10  # how much should we compress the x-axis (time)
start_freq = 300  # Hz # What frequency to start sampling our melS from
end_freq = 8000  # Hz # What frequency to stop sampling our melS from


def get_data(path_to_data):
    f = wave.open(path_to_data)
    channels = f.getnchannels()
    freq, data = wav.read(path_to_data)
    data = data.T[0].astype('float')
    return (freq, data)
    #if channels == 2:
    #    data = data.sum(axis=1) / 2
    #return (freq, data)

def read_params(feature_file_name):
    with open(feature_file_name, 'r') as ff:
        return json.load(ff)

def getPositionInSeconds(correlation, data, freq):
    return round((np.argmax(correlation) - len(data)/2)/(freq), 2)

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# Filter requirements.
order = 6
fs = 44.100       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz


class Predic_By_Following:

    def __init__(self, golden_data, golden_features, stream_features, bws = 4, sws = 2):
        self.big_window_size = bws * FREQ
        self.small_window_size = sws * FREQ
        self.gdata = golden_data
        self.golden_features = golden_features['points']
        self.stream_features = stream_features['points']
        self.curr_feature = 0
        self.curr_gold_second = bws #TODO DON't remember why we changed from 0 to bws
        self.stream_data = []
        self.stream_window_size = int(bws * 2 * FREQ)
        self.remained = []
        self.slice = 1
        self.latest_valid = 999999
        self.corr_thresh = self.stream_window_size * 0.4
        self.validation_count = 3
        self.allowed_incorrectness = 1
        self.gstart = 0
        self.gend = 0
        self.latest_stream_point = 0
        self.nearest_threshold = 3



    def compare(self, audio1, audio2):
        """ 
        returns the index in audio1 where audio2 is found.
        """
        #print("\t\t\t\tChonchurik = ", len(audio1), len(audio2))
        #audio1 = butter_lowpass_filter(audio1, cutoff, fs, order)
        #audio2 = butter_lowpass_filter(audio2, cutoff, fs, order)
        #audio1 = signal.detrend(audio1, type='constant')
        #audio2 = signal.detrend(audio2, type='constant')

        #s = sum(audio1)
        #audio1 = [float(i)/s for i in audio1]
        #s = sum(audio2)
        #audio2 = [float(i)/s for i in audio2]
        #print ("ARR1 ", len(audio1))
        #print (audio1[0:10])
        #print ("ARR2 ", len(audio2))
        #print (audio2[0:10])
        #print (type(audio1[0]))
        corr = signal.correlate(audio1, audio2, mode='same') #searches second array in first
        #print ("CORR ", len(corr))
        #print (corr[0:10])
        #corr = corr[int(len(audio2)/2) : int(len(audio2)/2 + len(audio1))]
        m = np.argmax(corr)
        res = (m - len(audio2)/2)
        s = np.sum(corr)
        log.debug ("Sum: {0},  Mean: {1}, Median: {2}, Peak: {3}, m: {4}".format(np.sum(corr), np.mean(corr), np.median(corr), np.max(corr), m))
        print ("INDEX : ", res)
        return (m, res)

    def is_valid_remained(self, corr_max):
        if len(self.remained) < 4:
            return False
        log.debug ("\t   : {}".format(self.remained[-self.validation_count:]))
        print("AAAAAAAAAAAAAAA: ", -self.validation_count, " --- ", len(self.remained), " :::::::: ", self.validation_count);
        tmp = [t - self.remained[-1] for t in self.remained[-self.validation_count:]]
        log.debug ("\tTMP: {}".format(tmp))
        thresh = 0
        l = len(tmp) - 1
        for i in range(0, l):
            if abs(tmp[i] - l + i) > 0.5:
                thresh +=1
            if thresh > self.allowed_incorrectness:
                return False
        return True



    def run(self, stream_snippet, current_stream_second):
        pass




parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", required=True, help="Input test WAV file.")
parser.add_argument("--test_features", required=True, help="Test file features.")
parser.add_argument("-g", "--golden", required=True, help="Input golden WAV file.")
#parser.add_argument("-c", "--config", required=True, help="Configuration parameters")
parser.add_argument("--golden_features", required=True, help="Golden file features.")
parser.add_argument("-o", "--output", help="Path to output directory to store the snippets of corresponding points.")
parser.add_argument("-d", "--debug", action='store_true', help="Enable debugging")

def generate_fft(fs, data):
    a = data.T[0] # this is a two channel soundtrack, I get the first track
    b=[(ele/2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
    c = rfft(b) # create a list of complex number
    return c


def ff(fs, x):
    segment_size = 512

    x = x / 32768.0  # scale signal to [-1.0 .. 1.0]

    noverlap = int(segment_size / 2)
    print (noverlap, segment_size)
    f, Pxx = signal.welch(x,                        # signal
                          fs=fs,                    # sample rate
                          nperseg=segment_size,     # segment size
                          window='hanning',         # window type to use
                          nfft=segment_size,        # num. of samples in FFT
                          detrend=False,            # remove DC part
                          scaling='spectrum',       # return power spectrum [V^2]
                          noverlap=noverlap)        # overlap between segments

# set 0 dB to energy of sine wave with maximum amplitude
    ref = (1/np.sqrt(2)**2)   # simply 0.5 ;)
    p = 10 * np.log10(Pxx/ref)

    fill_to = -150 * (np.ones_like(p))  # anything below -150dB is irrelevant
    plt.fill_between(f, p, fill_to )
    plt.xlim([f[2], f[-1]])
    plt.ylim([-90, 6])
# plt.xscale('log')   # uncomment if you want log scale on x-axis
    plt.xlabel('f, Hz')
    plt.ylabel('Power spectrum, dB')
    plt.grid(True)
    plt.show()




def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    print (nyq, low, high)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

if __name__ == '__main__':
    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)

    #test_freq, test_data = get_data(args.test)
    #golden_freq, golden_data = get_data(args.golden)
    test_features = read_params(args.test_features)
    golden_features = read_params(args.golden_features)
     
    gname = (os.path.splitext(os.path.basename(args.golden)))[0]
    tname = (os.path.splitext(os.path.basename(args.test)))[0]

    gs = golden_features['points'][0]['d']
    ts = test_features['points'][0]['d']

    #stream = test_data
    fname = tname
    expected_start = ts

    lowcut = 350
    highcut = 450
    window = 1
    start = int((gs) * FREQ)
    end = start + int(window *  FREQ)

    golden_data, sr = librosa.load(args.golden, sr=FREQ)
    stream_data, sr = librosa.load(args.test, sr=FREQ)

    g_cropped = golden_data[start:end]
    #g_cropped = butter_bandpass_filter(g_cropped, 350/FREQ, 450/FREQ, rate, order=1)

    l = 0
    h = 128
    n_mels = 128
    n_fft = 1024
    print (FREQ)
    hop_length = 128
    #g_cropped = normalize(g_cropped[:,np.newaxis], axis=0).ravel()
    #g_cropped = butter_bandpass_filter(g_cropped, 250/FREQ, 550/FREQ, rate, order=1)
    S = librosa.feature.melspectrogram(y=g_cropped, sr=FREQ, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    #S = librosa.feature.chroma_stft(y=g_cropped, sr=FREQ, tuning=0, norm=2, hop_length=hop_length, n_fft=n_fft)
    #S = librosa.feature.chroma_cqt(y=g_cropped, sr=FREQ, hop_length=hop_length)

    #S = librosa.feature.chroma_stft(y=g_cropped, sr=FREQ, n_fft=n_fft, hop_length=hop_length)
    #S = librosa.power_to_db(S, ref=np.max)
    S = np.log(0.1 + S)
    g_filtered = S[l:h]
    #g_filtered = librosa.util.normalize(g_filtered, threshold=80, fill=True)
    #plt.imshow(g_filtered, aspect=1)
    #plt.show()
    

    #start = int((ts - 10) * FREQ)
    #end = start + 2 *  FREQ
    #td = stream[start:end]

    print (FREQ)
    #stream_data = butter_bandpass_filter(stream_data, 250/FREQ, 550/FREQ, rate, order=1)
    #stream_data = normalize(stream_data[:,np.newaxis], axis=0).ravel()
    St = librosa.feature.melspectrogram(y=stream_data, sr=FREQ, hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
    #St = librosa.feature.chroma_stft(y=stream_data, sr=FREQ, tuning=0, norm=2, hop_length=hop_length, n_fft=n_fft)
    #St = librosa.feature.chroma_cqt(y=stream_data, sr=FREQ, hop_length=hop_length)
    #St = librosa.feature.chroma_stft(y=stream_data, sr=FREQ, n_fft=n_fft, hop_length=hop_length)
    #St = librosa.power_to_db(St, ref=np.max)
    St = np.log(0.1 + St)
    t_filtered = St#St[l:h]
    #t_filtered = librosa.util.normalize(t_filtered, threshold=80, fill=True)
    plt.imshow(t_filtered, aspect=1)
    plt.show()

    corr = signal.correlate(t_filtered, g_filtered, mode='same') #searches second array in first
    index = int(len(corr)/2)
    plt.plot(np.arange(len(corr[index])) * hop_length/FREQ, corr[index])
    #plt.plot(corr[0])
    #plt.imshow(corr, aspect=10)
    plt.show()
    
    print(expected_start)
    print(corr.argmax())


    img = t_filtered
    img2 = img.copy()
    template = g_filtered
    w, h = template.shape[::-1]
    plt.imshow(img, aspect=1)
    
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for meth in methods:
        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, [0,0,0], 2)

        plt.subplot(121),plt.imshow(g_filtered)
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(img)
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()



        plt.imshow(img, aspect=1)
        plt.show()

        exit()

    exit()






    #PBF = Predic_By_Following(golden_data, golden_features, test_features)

    #notification_points = []
    #for i in range(0, int(len(test_data)/FREQ)):
    #    log.debug ("\n====================================================")
    #    (notify, is_valid, remained_time) = PBF.run(test_data[i*FREQ : (i + 1) * FREQ], i)
    #    log.debug ("RESULT: {} {}, {}, {}, Predicted to notify at: {}".format(i, notify, is_valid, remained_time, i + remained_time))
    #    if notify:
    #        notification_points.append(i + remained_time)
    #    if len(notification_points) == len(test_features['points']):
    #        break;

    #for i in range (0, len(notification_points)):
    #    log.info ("\t{}: Expected point {}, \tnotified at {}, \tdelay is {}".format(args.test, round(test_features['points'][i]['p'], 5), round(notification_points[i], 5), round(notification_points[i] - test_features['points'][i]['p'],5)))
