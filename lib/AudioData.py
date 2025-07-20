import numpy as np
import pydub
import wave
import os

class AudioData:
    # default 3.072M = 48K/4bytes/2ch
    def __init__(self, data: np.ndarray, sample_rate: int, sample_width: int = 4, channels: int = 2):
        self.data = data
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

    @classmethod
    def from_file(cls, file_path: str):
        print(f"Load from: {file_path}")
        audio_segment = pydub.AudioSegment.from_file(file_path)
        # to do bit conversion
        data = np.array(audio_segment.get_array_of_samples())
        return cls(
            data=data,
            sample_rate=audio_segment.frame_rate,
            sample_width=audio_segment.sample_width,
            channels=audio_segment.channels
        )

    @classmethod
    def from_sine(cls, duration: float, freq: float = 1000, amp: float = -1,
                  rate: int = 48000, width: int = 4, channels: int = 2):
        print(f"Make sine wave: {rate}, {channels}ch, amp: {amp}, freq: {freq}Hz, {duration}s")
        t = np.linspace(0., duration, int(rate * duration), endpoint=False)
        nptype = [np.int8, np.int16, np.int32, np.int32][width-1]
        if(amp < 0):
            amp = np.iinfo(nptype).max * 0.9
        mono_sine = (amp * np.sin(2. * np.pi * freq * t)).astype(nptype)
        multi_sine = np.tile(mono_sine.reshape(-1, 1), (1, channels))
        data = multi_sine.flatten()
        return cls(data=data, sample_rate=rate, sample_width=width, channels=channels)

    @classmethod
    def from_multi_sine(cls, duration: float, freqs: list, amp: float = -1,
                        rate: int = 48000, width: int = 4, channels: int = 2):
        print(f"Make multi sine wave: {rate}, {channels}ch, amp: {amp}, freqs: {freqs}, {duration}s")
        t = np.linspace(0., duration, int(rate * duration), endpoint=False)
        nptype = [np.int8, np.int16, np.int32, np.int32][width-1]
        data = np.zeros(int(rate * duration * channels), dtype=np.int64)
        if(amp < 0):
            amp = np.iinfo(nptype).max * 0.9
        for freq in freqs:
            mono_sine = (amp * np.sin(2. * np.pi * freq * t)).astype(nptype)
            multi_sine = np.tile(mono_sine.reshape(-1, 1), (1, channels))
            data += multi_sine.flatten().astype(np.int64)
        data = (data // len(freqs)).astype(nptype)
        return cls(data=data, sample_rate=rate, sample_width=width, channels=channels)

    # mixing audio data
    def mix(self, audio_data):
        if self.sample_rate != audio_data.sample_rate or self.channels != audio_data.channels:
            raise ValueError("Sample rate and channels must match to merge audio data.")
        merged_data = self.data.copy().astype(np.int64) + audio_data.data.astype(np.int64)
        merged_data = merged_data//2
        self.data = np.clip(merged_data, self.min_amp, self.max_amp).astype(self.data.dtype)
        return self

    def save(self, file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with wave.open(file_path, 'w') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.data.tobytes())
        print(f"Saved file: {file_path}")
        return self

    def copy(self):
        return AudioData(
            data=self.data.copy(),
            sample_rate=self.sample_rate,
            sample_width=self.sample_width,
            channels=self.channels
        )

    def getData(self, ch=None):
        return self.data[ch::self.channels] if ch else self.data

    @property
    def max_val(self):
        """Maximum value based on data."""
        return self.data.max()
    @property
    def min_val(self):
        """Minimum value based on data."""
        return self.data.min()

    @property
    def max_amp(self):
        """Maximum amplitude based on sample width."""
        return 2**(self.sample_width * 8 - 1) - 1

    @property
    def min_amp(self):
        """Minimum amplitude based on sample width."""
        return -2**(self.sample_width * 8 - 1) + 1

    def apply(self, *args, **kwargs):
        func = args[0]
        args = args[1:]
        """
        Apply a function to the audio data.
        :param func: Function to apply to the audio data.
        :return: self with modified data.
        """
        self.data = func(self, *args, **kwargs)
        print(f"Applied function: {func.__name__} with args: {args}, kwargs: {kwargs}")
        return self