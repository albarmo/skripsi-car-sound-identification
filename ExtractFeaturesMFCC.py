import numpy
import scipy.io.wavfile
from scipy.fftpack import dct
import IPython.display as ipd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import get_window

# Class to handle all HMM related processing
class ExtractFeaturesMFCC():

    def __init__(self, model):
        self.model = model

    def show(self):
        print("Model is", self.model )
        # Reed data 
        filePath = '/Users/albarms/Desktop/Apps Source/Spoken-Digit-Recognition-master/new_spoken_digit/0_jackson_0.wav'
        ipd.Audio(filePath)

        # Get sample rate
        sample_rate, audio = wavfile.read(filePath)
        sample_rate, signal = scipy.io.wavfile.read(filePath)  # File assumed to be in the same directory
        signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds

        print("Sample rate: {0}Hz".format(sample_rate))
        print("Audio duration: {0}s".format(len(audio) / sample_rate))


        plt.figure(figsize=(15,4))
        plt.title('Frame Before Pre Emphasis')
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.grid(True)


        # Pre-Emphasis
        pre_emphasis = 0.97
        emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        plt.figure(figsize=(15,4))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(emphasized_signal)), emphasized_signal)
        plt.grid(True)

        # Framing
        frame_size = 0.025
        frame_stride = 0.01

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = numpy.zeros((pad_signal_length - signal_length))
        pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(numpy.int32, copy=False)]

        # print("Framed audio shape: {0}".format(frames))
        # print("First frame:")
        # frames[1]

        # plt.figure(figsize=(15,4))
        # plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(frames[0])), frames[0])
        # plt.grid(True)


        # Window
        ind = 10
        plt.figure(figsize=(15,6))
        plt.subplot(2, 1, 1)
        plt.plot(frames[ind])
        plt.title('Original Frame')
        plt.grid(True)

        frames *= numpy.hamming(frame_length)
        FFT_size = 1048


        plt.subplot(2, 1, 2)
        plt.plot(frames[ind])
        plt.title('Frame After Windowing')
        plt.grid(True)

        # print(frames)

        # Fourier-Transform and Power Spectrum
        NFFT = 512

        mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        print("Mag Frames :", mag_frames)
        print("Pow Frames :", pow_frames)


        # Filter Banks
        nfilt = 40

        low_freq_mel = 0
        high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = numpy.dot(pow_frames, fbank.T)
        filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * numpy.log10(filter_banks)  # dB


        # filters = get_filters(mel_points, 512)
        plt.figure(figsize=(15,4))
        for n in range(fbank.shape[0]):
            plt.plot(fbank[n])


        # Mel-frequency Cepstral Coefficients (MFCCs)
        num_ceps = 12
        cep_lifter = 22

        mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

        (nframes, ncoeff) = mfcc.shape
        n = numpy.arange(ncoeff)
        lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
        mfcc *= lift  #*

        plt.figure(figsize=(15,5))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.imshow(mfcc, aspect='auto', origin='lower');

        # Mean Normalization
        filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
        print("The mean-normalized filter banks : ")
        print(filter_banks[0])

        plt.figure(figsize=(15,5))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.imshow(filter_banks, aspect='auto', origin='lower');

        # The mean-normalized MFCCs:
        mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)

        return mfcc
      


    # plt.figure(figsize=(15,5))
    # plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
    # plt.imshow(filter_banks, aspect='auto', origin='lower');