import webrtcvad
import numpy as np
import matplotlib.pyplot as plt

class VAD:

    class Frame(object):
        """Represents a "frame" of audio data."""
        def __init__(self, bytes, timestamp, duration):
            self.bytes = bytes
            self.timestamp = timestamp
            self.duration = duration

    def __init__(self, fs=16000, frame_len=30,level=1):
        self.fs = fs
        self.frame_len = frame_len
        self.level = level
        self.vad_ins = webrtcvad.Vad(self.level)

    def frame_generator(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        sample_rate = self.fs
        frame_duration_ms = self.frame_len
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield self.Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def np_to_bytes(self, ip):
        # ip_norm = librosa.util.normalize(ip)
        ip_int = ip*32767
        ip_int16 = ip_int.astype(np.int16)
        ip_bytes = ip_int16.tobytes()
        return ip_bytes

    def per_frame(self, frame):
        is_speech = self.vad_ins.is_speech(frame.bytes, self.fs)
        if is_speech:
            return 1
        else:
            return 0

    def estimate(self,ip):
        data = self.np_to_bytes(ip)
        frames = self.frame_generator(data)
        frames = list(frames)
        vad_rec = [self.per_frame(frame) for frame in frames]
        per_speech = vad_rec.count(1)/len(vad_rec)
        return per_speech, vad_rec

    def vad_graph(self,ip, ip_org):
        _, vad_rec = self.estimate(ip)
        n_frame_sample = int(np.ceil(len(ip)/len(vad_rec)))
        vad_sig = np.array([0])
        for val in vad_rec:
            if val == 1:
                temp = np.ones((n_frame_sample,))*.5
                vad_sig = np.append(vad_sig,temp)
            else:
                temp = np.zeros((n_frame_sample,))
                vad_sig = np.append(vad_sig, temp)
        # x = np.linspace(0,(len(ip_org)/self.fs),len(ip_org))
        # print(len(x))
        # plt.figure()
        # plt.plot(ip_org)
        # plt.plot(vad_sig)
        return vad_sig




# def read_audio(address):
#     fs, ip = wavfile.read(address)
#     ip = ip/32768
#     ip_8ch = np.array(ip, dtype=np.float64)
#     ip_7ch = np.delete(ip_8ch, [7], axis=1)
#     print('Finished reading the audio file \n ')
#     return fs, ip_7ch
#
# fs, ip_7ch = read_audio('01_0.wav')
# ip = librosa.util.normalize(ip_7ch[0:5*fs,0])
# #
# vad_process = voiceactivitydetection(fs=fs, level=3)
# per_speech, vad_rec = vad_process.estimate(ip)

# def read_wave(path):
#     """Reads a .wav file.
#     Takes the path, and returns (PCM audio data, sample rate).
#     """
#     with contextlib.closing(wave.open(path, 'rb')) as wf:
#         num_channels = wf.getnchannels()
#         assert num_channels == 1
#         sample_width = wf.getsampwidth()
#         assert sample_width == 2
#         sample_rate = wf.getframerate()
#         assert sample_rate in (8000, 16000, 32000, 48000)
#         pcm_data = wf.readframes(wf.getnframes())
#         return pcm_data, sample_rate


# data = ip_bytes
# frame_len = 30
# level = int(1)
#
# vad = webrtcvad.Vad(level)
# frames = frame_generator(frame_len, data, fs)
# frames = list(frames)
#
# vad_record = []
# for i in range(len(frames)):
#     is_speech = vad.is_speech(frames[i].bytes, fs)
#     if is_speech:
#         vad_record.append(1)
#     else:
#         vad_record.append(0)

    # data, fs = read_wave('sample_1ch.wav')
    #
    # vad = webrtcvad.Vad(int(1))
    #
    # frames = frame_generator(30, data, fs)
    # frames = list(frames)
    #
    # vad_record = []
    # for i in range(len(frames)):
    #     is_speech = vad.is_speech(frames[i].bytes, fs)
    #     if is_speech:
    #         vad_record.append(1)
    #     else:
    #         vad_record.append(0)