import torch
from torchvision import transforms
import numpy as np
import librosa
import torchaudio


class Audio_Transform:

    def __init__(self, para, device):

        self.fs = para['fs']
        self.time= para['time']
        self.n_fft = para['n_fft']
        self.win_length = para['win_length']
        self.hop_length = para['hop_length']
        self.device = device
        ## Sri
        self.ldness = para['targetloudness']
        self.spectrum = self.get_spectrogram()
        self.spectrum = self.spectrum.to(self.device)
        self.platform = para["platform_name"]
        if self.platform == 'Windows':
            torchaudio.set_audio_backend(backend="soundfile")
        else:
            torchaudio.set_audio_backend(backend="sox_io")
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        self.amp_to_db = self.amp_to_db.to(self.device)

        self.au_to_img = transforms.Compose([transforms.ToPILImage(),
                                            transforms.ToTensor()])

    def digital_gain(audioData, targetLoudness):
            rms=(audioData**2).mean() ** 0.5
            
            if rms > 0.0:
                scalar = (10 ** (targetLoudness / 20) ) / rms 
                audioData = audioData * scalar

            return audioData

    def wav_to_spectrogram(self, ip):
        s = librosa.stft(ip,
                         n_fft= self.n_fft,
                         hop_length= self.hop_length,
                         win_length= self.win_length)
        s_log = np.abs(s)
        s_log = librosa.power_to_db(s_log, ref=np.max)

        return s_log
    
    def get_spectrogram(self):
        spectrum = torchaudio.transforms.Spectrogram(n_fft=self.n_fft,
                                                     win_length=self.win_length,
                                                     hop_length=self.hop_length,
                                                     normalized=True)
        return spectrum

    def normalize_spectra(self, x):

        min_val1 = np.min(x, axis=0)
        min_val2 = np.min(min_val1)

        x_step1 = x - min_val2

        max_val1 = np.max(x_step1, axis=0)
        max_val2 = np.max(max_val1)

        x_norm = x_step1 / max_val2

        return x_norm

    def normalize(self, ip):
        ip_norm = librosa.util.normalize(ip)
        return ip_norm

    def spec_to_img(self,spectra):
        img = None
        for i in range(spectra.shape[0]):
            if i == 0:
                temp = self.au_to_img(spectra[i, :, :])
                img = temp
            else:
                temp = self.au_to_img(spectra[i, :, :])
                img = torch.vstack((img, temp))
       
        return img

    def main(self, ip):
        ## Sri
        # ip = self.digital_gain(ip,self.ldness)
        # ip = torch.nan_to_num(ip)
        ip = self.normalize(ip)
        spectra = self.wav_to_spectrogram(ip)
        spectra_db = self.normalize_spectra(spectra)
        spectrum_db = torch.tensor(spectra_db)

        # spectra = self.spectrum(ip)
        # spectrum_db = self.amp_to_db(spectra)

        
        spectrum_img = self.au_to_img(spectrum_db)
        spectrum_img = spectrum_img.to(self.device, dtype=torch.float32)
        spectrum_img = spectrum_img[:,None,:,:]

        # spectra_th = torch.tensor(spectra_norm)
        # spectra_img = self.au_to_img(spectra_th)
        # spectra_img = spectra_img[None, :, :, :]
        # spectra_img = spectra_img.to(self.device, dtype=torch.float32)

        return spectrum_img


# class Audio_Transform:

#     def __init__(self, para, device):
#         self.fs = para['fs']
#         self.time= para['time']
#         self.n_fft = para['n_fft']
#         self.win_length = para['win_length']
#         self.hop_length = para['hop_length']
#         self.device = device
#         self.au_to_img = transforms.Compose([transforms.ToPILImage(),
#                                             transforms.ToTensor()])

#     def wav_to_spectrogram(self, ip):
#         s = librosa.stft(ip,
#                          n_fft= self.n_fft,
#                          hop_length= self.hop_length,
#                          win_length= self.win_length)
#         s_log = np.abs(s)
#         s_log = librosa.power_to_db(s_log, ref=np.max)

#         return s_log

#     def normalize_spectra(self, x):

#         min_val1 = np.min(x, axis=0)
#         min_val2 = np.min(min_val1)

#         x_step1 = x - min_val2

#         max_val1 = np.max(x_step1, axis=0)
#         max_val2 = np.max(max_val1)

#         x_norm = x_step1 / max_val2

#         return x_norm

#     def normalize(self, ip):
#         ip_norm = librosa.util.normalize(ip)
#         return ip_norm

#     def main(self, ip):

#         ip_norm = self.normalize(ip)
#         spectra = self.wav_to_spectrogram(ip_norm)
#         spectra_norm = self.normalize_spectra(spectra)
#         spectra_th = torch.tensor(spectra_norm)
#         spectra_img = self.au_to_img(spectra_th)
#         spectra_img = spectra_img[None, :, :, :]
#         spectra_img = spectra_img.to(self.device, dtype=torch.float32)

#         return spectra_img

















