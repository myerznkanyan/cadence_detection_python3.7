"""
Here is the documentation of this project


"""
import matplotlib
from PIL import Image

matplotlib.use('TKAgg')  # Specific for MAC OS
############### Import Libraries ###############
from matplotlib.mlab import window_hanning, specgram
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import PIL
import numpy as np
from keras.models import load_model
#import PySimpleGUI as sg
import time

############### Import Modules ###############

import mic_read


class note_detection:
    ############### Constants ###############

    SAMPLES_PER_FRAME = 2 # Number of mic reads concatenated within a single window
    nfft = 1024  # 256#1024 #NFFT value for spectrogram
    overlap = 1000  # 512 #overlap value for spectrogram
    rate = mic_read.RATE  # sampling rate
    checklist = [100]
    ############### Variables ###############
    bool = True
    prediction = 'note'
    start_time = time.time()

    ############### Functions ###############
    """
    get_sample:
    gets the audio data from the microphone
    inputs: audio stream and PyAudio object
    outputs: int16 array
    """

    def get_sample(self, stream, pa):
        data = mic_read.get_data(stream, pa)
        return data

    """
    get_specgram:
    takes the FFT to create a spectrogram of the given audio signal
    input: audio signal, sampling rate
    output: 2D Spectrogram Array, Frequency Array, Bin Array
    see matplotlib.mlab.specgram documentation for help
    """

    def get_specgram(self, signal, rate):
        arr2D, freqs, bins = specgram(signal, window=window_hanning,
                                      Fs=rate, NFFT=self.nfft, noverlap=self.overlap)
        return arr2D, freqs, bins
    """
    save_image:
    takes matplotlib object and saves the image in test_images directory using time inside name 
    to avoid redundancies
    inputs: matplolib_object
    outputs: none 
    """
    def save_img(self, matplotlib_obj):
        timestr = time.strftime("%Y.%m.%d-%H.%M.%S")
        matplotlib_obj.savefig('Data_sets/test_images/Note' + timestr + '.png')



    """
    predict:
    predicts the cadence based on model 
    input: CNN model
    output: 0 or 1 
    """

    def show_predicition(self, result):
        #TODO
        #GUI here
        if (result[0] == 0):
            print("Cadence")
        else:
            print("noCadence")

        #return np.array(pil_image)



    def matplotlib_to_numpy(self, fig):
        fig.canvas.draw()
        pil_image = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                        fig.canvas.renderer.tostring_rgb())
        print("Yes Works")

        return np.array(pil_image)



    def update_fig(self, n, stream, pa, fig, model, window, im):

        image = Image.open('cadense_image.png')
        print("Start Update Fig")
        data = self.get_sample(stream, pa)
        arr2D, freqs, bins = self.get_specgram(data, self.rate)

        im_data = im.get_array()
        arr2D[arr2D < 0.9] = 0.01
        if n < self.SAMPLES_PER_FRAME:
            im_data = np.hstack((im_data, arr2D))
            im.set_array(im_data)
        else:
            keep_block = arr2D.shape[1] * (self.SAMPLES_PER_FRAME - 1)
            im_data = np.delete(im_data, np.s_[:-keep_block], 1)
            im_data = np.hstack((im_data, arr2D))
            im.set_array(im_data)

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

       
        #[sg.Image(r'C:\PySimpleGUI\Logos\PySimpleGUI_Logo_320.png')]
        """
        delta = round((time.time() - self.start_time) * 1000)
        event, values = window.read(timeout=100)
        #window['_Note_'].update(value=self.prediction)
        window['_Note_'].update(value= sg.Image(r'C:\PySimpleGUI\Logos\PySimpleGUI_Logo_320.png'))
        window['stopwatch'].update(value=f'{delta//1000//60:02d}:{delta//1000%60:02d}:{delta%1000:03d}')
        if event is None or event == 'Exit':
            self.bool = False
            
        """
        return im,

    def start(self):
        ############### Initialize Plot ###############
        # Set up figure size
        fig = plt.figure(figsize=(0.64, 0.64))

        # Load CNN .h5 model
        model = load_model('Model/Spectogram_15|09|2020_v2.h5')

        # Launch the stream and the original spectrogram
        stream, pa = mic_read.open_mic()
        data = self.get_sample(stream, pa)
        arr2D, freqs, bins = self.get_specgram(data, self.rate)

        # Setup the plot paramters
        extent = (bins[0], bins[-1] * self.SAMPLES_PER_FRAME, freqs[-150], freqs[20])
        im = plt.imshow(arr2D, aspect='auto', extent=extent, interpolation="none",
                        cmap='gray', norm=LogNorm(vmin=.01, vmax=1))

        # get clear matplotlib spectogram
        plt.gca().invert_yaxis()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.text(2.85, 2.9, 'label',
                 bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 10})

        # Setup PySimpleGUI layout and window
        sg.theme('DarkAmber')

        layout = [[sg.Text(self.prediction, size=(10, 1), font=('Helvetica', 36), text_color='white', key='_Note_')],
                  [sg.Text("stopwatch", font=('Helvetica', 16), text_color='white', key='stopwatch')],
                  [sg.Button('Exit', size=(5, 1), font=('Helvetica', 16))]]

        window = sg.Window('Note detector', layout, size=(320, 320))

        # Continously calls update_fig function to get new spectrogram
        count = -1
        try:
            while self.bool:
                a += 1
                # Update function requires integer input
                count += 1
                # Update figue
                self.update_fig(count, stream, pa, fig, model, window, im)
                # Convert matplotlib Object to 3D numpy Array
                img_pred = np.expand_dims(self.matplotlib_to_numpy(fig), axis=0)
                # Predict based on model
                rslt = np.argmax(model.predict(img_pred), axis=1)
                # Show predicition result
                self.show_predicition(rslt)
                #print(a)

        except KeyboardInterrupt:
            # close the stream
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("Program Terminated")
            pass



