# Music_note_detection_python3.7

Cadence detection is implemented using CNN spectrogram classification.

Steps of the project

1. Microphone records sound using PyAudio
2. The audio array is filtered
3. After filtering FFT is computed
4. FFT 2D NUmpy Array is passed to matplotlib which creates 3d (64,64,3) image
5. An image converted to 3d NumPY array using Pillow
5. Loaded model predicts using 3d Numpy array

Conclusions after tests

* Using only one performer's data successes detecting Cadence part during test procedure
and succeeds for detecting wrong cadences before or after
* Turning on cadence detection 30 seconds earlier reduces the probability of failure
* Using grayscale images reduces the image size which accelerates classification
* Using several performer's data failed detecting Cadence
- There are several reasons for this failure
       ⋅⋅1. There are no identical Cadences. Each cadence has it's features and similarities
       ..2. Some similar parts of Cadence can be found before or after cadence
       ..3. The sequence is not considered in this project
* To increase the efficiency of CNN audio data is filtered  gain, volume and bandwidth
* High volume of sound increases the noise in Microphone
* Librosa mel-spectogram failed to plot differentiable spectrogram
* Instead of librosa matplotlib specgram was used for nfft and plotting 


### All collected spectrogram images and trained models are available.



