"""
run_specgram.py
Created By Alexander Yared (akyared@gmail.com)
Main Script for the Live Spectrogram project, a real time spectrogram
visualization tool
Dependencies: matplotlib, numpy and the mic_read.py module
"""
############### Import Libraries ###############
from matplotlib.mlab import window_hanning, specgram
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import numpy as np
import time
import random # for debugging

############### Import Modules ###############
import mic_read

############### Constants ###############
# SAMPLES_PER_FRAME = 10 #Number of mic reads concatenated within a single window
SAMPLES_PER_FRAME = 2
nfft = 1024  # 256#1024 #NFFT value for spectrogram
overlap = 1000 # 512 #overlap value for spectrogram
rate = mic_read.RATE  # sampling rate

############### Functions ###############
"""
get_sample:
gets the audio data from the microphone
inputs: audio stream and PyAudio object
outputs: int16 array
"""


def get_sample(stream, pa):
    data = mic_read.get_data(stream, pa)
    print("Inside get_sample")
    return data


"""
get_specgram:
takes the FFT to create a spectrogram of the given audio signal
input: audio signal, sampling rate
output: 2D Spectrogram Array, Frequency Array, Bin Array
see matplotlib.mlab.specgram documentation for help
"""


def get_specgram(signal, rate):
    arr2D, freqs, bins = specgram(signal, window=window_hanning,
                                  Fs=rate, NFFT=nfft, noverlap=overlap)
    print("Inside get_spectogram")
    return arr2D, freqs, bins


"""
update_fig:
updates the image, just adds on samples at the start until the maximum size is
reached, at which point it 'scrolls' horizontally by determining how much of the
data needs to stay, shifting it left, and appending the new data. 
inputs: iteration number
outputs: updated image
"""

def set_gain(datalist, numpy_gain):
    print("Max number in array", np.amax(datalist))
    print("Min number in array", np.amin(datalist))
    #ask = (datalist > -400)
    #datalist = datalist[mask]
    #mask = (datalist > 400) | (datalist > 0)
    #newdatalist = datalist[(datalist > 400) & (datalist <-400)]
    #datalist = np.where(datalist<400 and datalist > -400)
    #datalist[datalist < 400 and datalist > -400 ] = 40

    for i in range(len(datalist)):
        if datalist[i] < 200 and datalist[i] > 0:
            datalist[i] = random.randint(0,2)
        elif datalist[i] < 0 and datalist[i] > -200:
            datalist[i] = random.randint(-2, 0)
        #datalist[i] = 0

    print("##########")
    print("Max number in array", np.amax(datalist))
    print("Min number in array", np.amin(datalist))

    return datalist










def update_volume(datalist, volume):
    """ Change value of list of audio chunks """
    sound_level = (volume / 100.)
    print("inside update_voilume")
    for i in range(len(datalist)):
        chunk = np.fromstring(datalist[i], np.int16)

        chunk = chunk * sound_level

        datalist[i] = chunk.astype(np.int16)

    return datalist


def update_fig(n):
    data = get_sample(stream, pa)
    print("Mic data ", data)
    #print(data.shape)
    data_updated = set_gain(data, 10)
    data_updated = update_volume(data_updated, 10)
    #data_updated = set_gain(data_updated, 10)
    #set_gain(data,10)
    print("Mic data updated : ",data_updated)
    #print("n :",n)
    arr2D, freqs, bins = get_specgram(data_updated, rate)
    im_data = im.get_array()
    arr2D[arr2D < 0.99] = 0.01
    #print(arr2D)
    # print(type(n))
    # print(n)
    #print("Samples per frame", SAMPLES_PER_FRAME)
    if n < SAMPLES_PER_FRAME:
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)
    else:
        keep_block = arr2D.shape[1] * (SAMPLES_PER_FRAME - 1)
        im_data = np.delete(im_data, np.s_[:-keep_block], 1)
        im_data = np.hstack((im_data, arr2D))
        im.set_array(im_data)
    #plt.gca().set_axis_off()
    #plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                    hspace=0, wspace=0)
    #plt.margins(0, 0)
    #plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #plt.gca().yaxis.set_major_locator(plt.NullLocator())
    """ try:
        plt.savefig('Photos/gray.jpeg')
    except:
        print('')
    """
    return im,


if __name__ == "__main__":
    ############### Initialize Plot ###############
    fig = plt.figure()
    #real sizes
    #fig = plt.figure()
    """
    Launch the stream and the original spectrogram
    """
    stream, pa = mic_read.open_mic()
    data = get_sample(stream, pa)
    arr2D, freqs, bins = get_specgram(data, rate)
    """
    Setup the plot paramters
    """
    extent = (bins[0], bins[-1] * SAMPLES_PER_FRAME, freqs[-200], freqs[0])

    im = plt.imshow(arr2D, aspect='auto', extent=extent, interpolation="none",
                    cmap='gray', norm=LogNorm(vmin=.01, vmax=1))
    #uncomment for plotting
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Real Time Spectogram')
    plt.gca().invert_yaxis()

    # uncomment for data collection
    #plt.gca().axes.get_xaxis().set_visible(False)
    #plt.gca().axes.get_yaxis().set_visible(False)
    #plt.text(2.85, 2.9, 'label',
    #   bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 10})
    plt.colorbar() #enable if you want to display a color bar
    #plt.savefig('Photos/books_read' + timestr + '.png')

    ############### Animate ###############
    anim = animation.FuncAnimation(fig, update_fig, blit=False,interval = mic_read.CHUNK_SIZE / 1000)
    count = -1

    try:
         plt.show()
    except:
         print("Plot Closed")

    ############### Terminate ###############
    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Program Terminated")