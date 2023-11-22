import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import librosa
# import time
# from datetime import timedelta as td


class Enhance_Boost:

    def __init__(self,noise_decrease=0.90, ref_noise=False, ref_noise_address=None):
        self.noise_decrease = noise_decrease
        self.ref_noise = ref_noise
        self.ref_noise_address = ref_noise_address

    def _stft(self, y, n_fft, hop_length, win_length):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def _istft(self, y, hop_length, win_length):
        return librosa.istft(y, hop_length, win_length)


    def _amp_to_db(self, x):
        return librosa.core.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)


    def _db_to_amp(self, x,):
        return librosa.core.db_to_amplitude(x, ref=1.0)


    def plot_spectrogram(self, signal, title):
        fig, ax = plt.subplots(figsize=(20, 4))
        cax = ax.matshow(
            signal,
            origin="lower",
            aspect="auto",
            cmap=plt.cm.seismic,
            vmin=-1 * np.max(np.abs(signal)),
            vmax=np.max(np.abs(signal)),
        )
        fig.colorbar(cax)
        ax.set_title(title)
        plt.tight_layout()
        plt.show()


    def plot_statistics_and_filter(self, mean_freq_noise,
                                   std_freq_noise,
                                   noise_thresh,
                                   smoothing_filter):
        fig, ax = plt.subplots(ncols=2, figsize=(20, 4))
        plt_mean, = ax[0].plot(mean_freq_noise, label="Mean power of noise")
        plt_std, = ax[0].plot(std_freq_noise, label="Std. power of noise")
        plt_std, = ax[0].plot(noise_thresh, label="Noise threshold (by frequency)")
        ax[0].set_title("Threshold for mask")
        ax[0].legend()
        cax = ax[1].matshow(smoothing_filter, origin="lower")
        fig.colorbar(cax)
        ax[1].set_title("Filter for smoothing Mask")
        plt.show()


    def removeNoise(self, audio_clip, noise_clip):
        """Remove noise from audio based upon a clip containing only noise

        Args:
            audio_clip (array): The first parameter.
            noise_clip (array): The second parameter.
            n_grad_freq (int): how many frequency channels to smooth over with the mask.
            n_grad_time (int): how many time channels to smooth over with the mask.
            n_fft (int): number audio of frames between STFT columns.
            win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
            hop_length (int):number audio of frames between STFT columns.
            n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
            prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
            visual (bool): Whether to plot the steps of the algorithm

        Returns:
            array: The recovered signal with noise subtracted

        """
        n_grad_freq = 2
        n_grad_time = 4
        n_fft = 2048
        win_length = 2048
        hop_length = 512
        n_std_thresh = 1.5
        prop_decrease = self.noise_decrease
        # verbose = False,
        # visual = False

        # if verbose:
        #     start = time.time()
        # STFT over noise
        noise_stft = self._stft(noise_clip, n_fft, hop_length, win_length)
        noise_stft_db = self._amp_to_db(np.abs(noise_stft))  # convert to dB
        # Calculate statistics over noise
        mean_freq_noise = np.mean(noise_stft_db, axis=1)
        std_freq_noise = np.std(noise_stft_db, axis=1)
        noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
        # if verbose:
        #     print("STFT on noise:", td(seconds=time.time() - start))
        #     start = time.time()
        # STFT over signal
        # if verbose:
        #     start = time.time()
        sig_stft = self._stft(audio_clip, n_fft, hop_length, win_length)
        sig_stft_db = self._amp_to_db(np.abs(sig_stft))
        # if verbose:
        #     print("STFT on signal:", td(seconds=time.time() - start))
        #     start = time.time()
        # Calculate value to mask dB to
        mask_gain_dB = np.min(self._amp_to_db(np.abs(sig_stft)))
        # print(noise_thresh, mask_gain_dB)
        # Create a smoothing filter for the mask in time and frequency
        smoothing_filter = np.outer(
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_freq + 2),
                ]
            )[1:-1],
            np.concatenate(
                [
                    np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                    np.linspace(1, 0, n_grad_time + 2),
                ]
            )[1:-1],
        )
        smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
        # calculate the threshold for each frequency/time bin
        db_thresh = np.repeat(
            np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
            np.shape(sig_stft_db)[1],
            axis=0,
        ).T
        # mask if the signal is above the threshold
        sig_mask = sig_stft_db < db_thresh
        # if verbose:
        #     print("Masking:", td(seconds=time.time() - start))
        #     start = time.time()
        # convolve the mask with a smoothing filter
        sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
        sig_mask = sig_mask * prop_decrease
        # if verbose:
        #     print("Mask convolution:", td(seconds=time.time() - start))
        #     start = time.time()
        # mask the signal
        sig_stft_db_masked = (
            sig_stft_db * (1 - sig_mask)
            + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask
        )  # mask real
        sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
        sig_stft_amp = (self._db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (
            1j * sig_imag_masked
        )
        # if verbose:
        #     print("Mask application:", td(seconds=time.time() - start))
        #     start = time.time()
        # recover the signal
        recovered_signal = self._istft(sig_stft_amp, hop_length, win_length)
        recovered_spec = self._amp_to_db(
            np.abs(self._stft(recovered_signal, n_fft, hop_length, win_length))
        )
        # if verbose:
        #     print("Signal recovery:", td(seconds=time.time() - start))
        # if visual:
        #     plot_spectrogram(noise_stft_db, title="Noise")
        # if visual:
        #     plot_statistics_and_filter(
        #         mean_freq_noise, std_freq_noise, noise_thresh, smoothing_filter
        #     )
        # if visual:
        #     plot_spectrogram(sig_stft_db, title="Signal")
        # if visual:
        #     plot_spectrogram(sig_mask, title="Mask applied")
        # if visual:
        #     plot_spectrogram(sig_stft_db_masked, title="Masked signal")
        # if visual:
        #     plot_spectrogram(recovered_spec, title="Recovered spectrogram")
        return recovered_signal

    def enhance(self, audio_clip):
        if self.ref_noise is True:
            noise_clip, _ = librosa.load(self.ref_noise_address, sr= None,mono=True)
            noise_clip = librosa.util.normalize(noise_clip)
        else:
            noise_clip = audio_clip
        enhance_ip = self.removeNoise(audio_clip,noise_clip)
        return enhance_ip

# import soundfile as sf
#
# ip, fs = librosa.load('./data/data_21-10/02.wav', sr=None, mono=True)
# ip = librosa.util.normalize(ip)
# output = removeNoise(audio_clip=ip, noise_clip=ip, prop_decrease=0.90)
# sf.write('out_audacity.wav',output,fs)