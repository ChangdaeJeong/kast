import numpy as np
from scipy.signal import butter, lfilter

class AudioNoise:
    """AudioFilter class for applying various filters to audio data.
    Provides methods to add DC offset, clipping, and pop noise to audio data.
    """
    @staticmethod
    def dc_offset(audio_data, offset: int):
        """Add DC offset to the audio data."""
        ndata = audio_data.data.copy()
        ndata = np.clip(ndata.astype(np.float32) + offset,
                        audio_data.min_amp, audio_data.max_amp).astype(ndata.dtype)
        print(f"  - Add DC offset (Level: {offset})")
        return ndata

    @staticmethod
    def clipping(audio_data, multiple: float):
        ndata = audio_data.data.copy()
        clipped_data = ndata.astype(np.float32) * multiple
        ndata = np.clip(clipped_data, audio_data.min_amp, audio_data.max_amp).astype(ndata.dtype)
        print(f"  - Make a clipping (Gain: {multiple})")
        return ndata

    @staticmethod
    def pop_noise(audio_data, target_sec: float, target_channel = [0],
                      noise_level: int = None, noise_duration_samples: int = 5):
        if noise_level is None:
            noise_level = audio_data.max_amp

        ndata = audio_data.data.copy()
        pop_position = int(target_sec * audio_data.sample_rate)
        for channel_idx in target_channel:
            for i in range(noise_duration_samples):
                ndata[(pop_position + i)* audio_data.channels + channel_idx] = \
                    np.clip(ndata[(pop_position + i) * audio_data.channels + channel_idx] + noise_level,
                            audio_data.min_amp, audio_data.max_amp)
            print(f"  - Add pop noise({target_sec}s)")
        else:
            print(f"  - Failed to add pop")
        return ndata

    @staticmethod
    def cut_noise(audio_data, target_sec: float, target_channel = [0],
                  noise_duration_samples: int = 5):
        ndata = audio_data.data.copy()
        cut_position = int(target_sec * audio_data.sample_rate)
        mono_data_len = len(ndata) // audio_data.channels
        for cur_position in range(cut_position, mono_data_len - noise_duration_samples):
            for channel_idx in target_channel:
                ndata[cur_position * audio_data.channels + channel_idx] = \
                    ndata[(cur_position + noise_duration_samples) * audio_data.channels + channel_idx]
        for last_position in range(mono_data_len - noise_duration_samples, mono_data_len):
            for channel_idx in target_channel:
                ndata[last_position * audio_data.channels + channel_idx] = 0
        return ndata

    @staticmethod
    def normalized_noise(audio_data, noise_level: int = 1000):
        """Add normalized noise to the audio data."""
        ndata = audio_data.data.copy().astype(np.float32)
        noise = np.random.normal(0, noise_level, ndata.shape).astype(ndata.dtype)
        ndata = np.clip(ndata + noise, audio_data.min_amp, audio_data.max_amp).astype(audio_data.data.dtype)
        print(f"  - Add normalized noise (Level: {noise_level})")
        return ndata

    @staticmethod
    def gaussian_noise(audio_data, noise_level: int = 1000):
        """Add Gaussian noise to the audio data."""
        ndata = audio_data.data.copy().astype(np.float32)
        noise = np.random.normal(0, noise_level, ndata.shape).astype(ndata.dtype)
        ndata = np.clip(ndata + noise, audio_data.min_amp, audio_data.max_amp).astype(audio_data.data.dtype)
        print(f"  - Add Gaussian noise (Level: {noise_level})")
        return ndata

class AudioFilter:
    @staticmethod
    def freq_pass_filter(audio_data, cutoff_freq: float, filter_type: str = 'low'):
        """Apply a pass filter to the audio data."""
        nyquist = 0.5 * audio_data.sample_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normal_cutoff, btype=filter_type, analog=False)
        deinterleaved_data = audio_data.data.reshape(-1, audio_data.channels)
        filtered_deinterleaved = np.apply_along_axis(lambda x: lfilter(b, a, x), 0, deinterleaved_data)
        print(f"  - {filter_type.capitalize()} filter applied (Cutoff frequency: {cutoff_freq}Hz)")
        return filtered_deinterleaved.flatten().astype(audio_data.data.dtype)

    @staticmethod
    def band_pass_filter(audio_data, low_cutoff: float, high_cutoff: float):
        """Apply a band-pass filter to the audio data."""
        nyquist = 0.5 * audio_data.sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(4, [low, high], btype='band', analog=False)
        deinterleaved_data = audio_data.data.reshape(-1, audio_data.channels)
        filtered_deinterleaved = np.apply_along_axis(lambda x: lfilter(b, a, x), 0, deinterleaved_data)
        print(f"  - Band-pass filter applied (Low: {low_cutoff}Hz, High: {high_cutoff}Hz)")
        return filtered_deinterleaved.flatten().astype(audio_data.data.dtype)

    @staticmethod
    def band_stop_filter(audio_data, low_cutoff: float, high_cutoff: float):
        """Apply a band-stop filter to the audio data."""
        nyquist = 0.5 * audio_data.sample_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = butter(4, [low, high], btype='bandstop', analog=False)
        deinterleaved_data = audio_data.data.reshape(-1, audio_data.channels)
        filtered_deinterleaved = np.apply_along_axis(lambda x: lfilter(b, a, x), 0, deinterleaved_data)
        print(f"  - Band-stop filter applied (Low: {low_cutoff}Hz, High: {high_cutoff}Hz)")
        return filtered_deinterleaved.flatten().astype(audio_data.data.dtype)