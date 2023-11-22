import numpy as np
from pyroomacoustics.transform.stft import analysis
from pyroomacoustics.utilities import highpass
from pyroomacoustics.doa.music import MUSIC

class Audio_ASL(object):

    def __init__(self, fs=16000):
        self.fs = fs
        self.c =  343
        self.freq_range = [2500.,4500.]
        self.nfft = 256
        self.stft_win = np.hanning(self.nfft)
        self.mode = 'far'
        self.array_geo, self.grid_azimuth = self.__array_geometry()
        self.scaling = 0.0004773295859045337

    def __array_geometry(self):
        grid_start_angle = 0.0  # angle from where grid search will startls
        grid_end_angle = 360.0  # angle where grid search will end
        grid_resolution = 100  # grid resolution

        array_geo = np.array(
            [[0, 0], [0, 0.0450], [0.0390, 0.0225], [0.0390, -0.0225], [0, -0.0450], [-0.0390, -0.0225],
             [-0.0390, 0.0225]])
        
        # array_geo = np.array(
        #     [[-0.0390,-0.0225],
        #     [0.0390,-0.0225]]
        # )

        array_geo = np.transpose(array_geo)

        grid_azimuth = np.linspace(grid_start_angle, grid_end_angle, grid_resolution) * np.pi / 180

        return array_geo, grid_azimuth

    def __stft(self,ins_ip):
        " Remove the DC bias using highpass filter "
        for s in ins_ip.T:
            s[:] = highpass(s, self.fs, 100.)
        ins_ip *= self.scaling

        " Compute the STFT frames needed "
        mic_signals = []
        for i in range(ins_ip.shape[1]):
            sig = np.array(ins_ip[:, i], dtype=np.float32)
            mic_signals.append(sig)
        x = np.array([
            analysis(signal, self.nfft, self.nfft // 2, win=self.stft_win, ).T
            for signal in mic_signals])

        return x

    def __asl_algo(self, num_src):
        asl_algo = MUSIC(self.array_geo,
                         self.fs, self.nfft, c= self.c,
                         num_src=num_src, azimuth=self.grid_azimuth, mode=self.mode)
        return asl_algo

    def localise(self,ip,num_src):

        x = self.__stft(ip)
        asl_algo = self.__asl_algo(num_src)
        asl_algo.locate_sources(x, freq_range=self.freq_range)
        est_position = asl_algo.azimuth_recon / np.pi * 180.
        est_position = np.round(np.sort(est_position))

        return est_position


