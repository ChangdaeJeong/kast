import numpy as np
from scipy.signal import correlate, coherence
from lib.AudioData import AudioData

class AudioEvaluator:
    """
    AudioEvaluator class for evaluating the similarity between original and recorded audio data.
    Provides methods to align signals, calculate mean squared error, and spectral coherence.
    """

    @staticmethod
    def _align_and_truncate(orig_data: np.ndarray, rec_data: np.ndarray, rate: int):
        """
        Cross-correlate the original and recorded audio data to find the optimal alignment point.
        Args:
            orig_data (np.ndarray): Original audio data.
            rec_data (np.ndarray): Recorded audio data.
            rate (int): Sample rate of the audio data.

        Returns:
            Tuple[np.ndarray, np.ndarray, int, float]
        """

        print("  - Normalize audio data and calculate cross-correlation...")
        # Normalize the audio data
        orig_norm = (orig_data - np.mean(orig_data)) / (np.std(orig_data) * len(orig_data))
        rec_norm = (rec_data - np.mean(rec_data)) / np.std(rec_data)

        # Calculate cross-correlation
        correlation = correlate(orig_norm, rec_norm, mode='full')
        # Find the lag that maximizes the correlation
        lag_in_samples = np.argmax(correlation) - (len(rec_data) - 1)
        max_corr = np.max(correlation)

        print(f"  - Time Lag: {lag_in_samples} sample ({lag_in_samples/rate:.4f}s)")

        # Align the signals based on the lag
        if lag_in_samples > 0:  # If original starts later than recorded
            start_index = lag_in_samples
            end_index = min(len(orig_data), len(rec_data) + lag_in_samples)
            aligned_orig = orig_data[start_index:end_index]
            aligned_rec = rec_data[0:end_index - start_index]
        else:  # If recorded starts later than original (or they start simultaneously)
            start_index = -lag_in_samples
            end_index = min(len(rec_data), len(orig_data) - lag_in_samples)
            aligned_rec = rec_data[start_index:end_index]
            aligned_orig = orig_data[0:end_index - start_index]

        # Make sure both arrays are of the same length
        min_len = min(len(aligned_orig), len(aligned_rec))
        aligned_orig = aligned_orig[:min_len]
        aligned_rec = aligned_rec[:min_len]

        return aligned_orig, aligned_rec, lag_in_samples, max_corr

    @staticmethod
    def evaluate(original: AudioData, recorded: AudioData):
        """
        Measure the similarity between original and recorded audio.

        Args:
            original (AudioData): Original audio data object.
            recorded (AudioData): Recorded audio data object.

        Returns:
            A dictionary containing similarity metrics for each channel.
        """
        if original.channels != recorded.channels:
            raise ValueError("channels of original and recorded audio must match.")

        results = {}
        print("\n[AudioEvaluator] Audio similarity evaluation started...")
        for ch in range(original.channels):
            print(f"===== Evaluating channel {ch} =====")
            orig_ch_data = original.getData(ch=ch).astype(np.float64)
            rec_ch_data = recorded.getData(ch=ch).astype(np.float64)

            # 1. Align the signals and get the peak correlation value
            aligned_orig, aligned_rec, lag, peak_corr = AudioEvaluator._align_and_truncate(orig_ch_data, rec_ch_data, original.sample_rate)

            # 2. Calculate Mean Squared Error (MSE)
            mse = np.mean((aligned_orig - aligned_rec) ** 2)

            # 3. Calculate Spectral Coherence
            # Measure how consistent the phase relationship is between the two signals across frequency bands
            nperseg = min(len(aligned_orig), 2048) # FFT segment size
            if nperseg > 0:
                f, Cxy = coherence(aligned_orig, aligned_rec, fs=original.sample_rate, nperseg=nperseg)
                avg_coherence = np.mean(Cxy)
            else:
                avg_coherence = 0.0

            results[f'channel_{ch}'] = {
                'lag_samples': lag,
                'peak_cross_correlation': peak_corr,
                'mean_squared_error': mse,
                'average_spectral_coherence': avg_coherence
            }
        print("\n[AudioEvaluator] Done.")
        return results