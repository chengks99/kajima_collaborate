from scipy.io import wavfile
import numpy as np
import librosa

def read_audio(address):
    fs, ip = wavfile.read(address)
    ip = ip / 32768
    ip = np.array(ip, dtype=np.float32)
    return fs, ip

class Sound_Level:

    def __init__(self, para):
        self.n_fft = para['n_fft']
        self.win_length = para['win_length']
        self.hop_length = para['hop_length']

    def wav_to_spectrogram(self, ip):
        s = librosa.stft(ip,
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         win_length=self.win_length)
        s_log = np.abs(s)
        s_log = librosa.power_to_db(s_log, ref=np.max)
        return np.flipud(s_log)

    def normalize_spectra(self, x):

        min_val1 = np.min(x, axis=0)
        min_val2 = np.min(min_val1)

        x_step1 = x - min_val2

        return x_step1

    def feature_variation(self, s_log):
        sum_feature = np.sum(s_log, axis=0)/s_log.shape[0]
        level = np.mean(sum_feature)
        return level

    def main(self, ip):
        s_log = self.wav_to_spectrogram(ip)
        spectra_norm = self.normalize_spectra(s_log)
        sound_level = self.feature_variation(spectra_norm)
        return sound_level

# para = {}
# para['n_fft'] = 1024
# para['win_length'] = int(0.05 * 16000)
# para['hop_length'] = int(0.02 * 16000)
# fs, ip = read_audio('./archive/0.wav')
#
# audio_level = Sound_Level(para)
# print(audio_level.main(ip[0:fs]))
