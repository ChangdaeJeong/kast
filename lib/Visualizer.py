import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class Visualizer:
    """
    Visualizer class for plotting audio data.
    Provides methods to plot waveforms and spectrograms of audio files.
    """
    @staticmethod
    def plot_wave(fig, axs, data, rate, color, label, alpha=0.5, title="", xlabel="", ylabel=""):
        time = np.arange(len(data))/rate
        axs.plot(time, data, color, label=label, alpha=alpha)
        if(title): axs.set_title(title)
        if(xlabel): axs.set_xlabel(xlabel)
        if(ylabel): axs.set_ylabel(ylabel)
        axs.legend()
        axs.grid(True)
        axs.set_xlim(0, len(data)/rate)

    @staticmethod
    def plot_spectrum(fig, axs, data, rate, title="", xlabel="", ylabel=""):
        f, t, Sxx = signal.spectrogram(data, rate)
        im1 = axs.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-9), shading='gouraud', cmap='magma')
        if(title): axs.set_title(title)
        if(xlabel): axs.set_xlabel(xlabel)
        if(ylabel): axs.set_ylabel(ylabel)
        axs.set_yscale('log') # Make y-axis logarithmic
        axs.set_ylim(10, rate / 2) # Show from 10Hz
        fig.colorbar(im1, ax=axs, format='%+2.0f dB', label='Intensity [dB]')

    @staticmethod
    def plot_audio_data(path, orig, rec):
        """Plot wave form and spectrum of audio data."""
        print(f"\n[Visualizer] plot '{path}' as file...")
        
        # wave, orig spectrum, rec spectrum
        n_plot = orig.channels + \
                 orig.channels + \
                 rec.channels

        fig, axs = plt.subplots(n_plot, 1, figsize=(15, 12))
        fig.suptitle('Original vs. Recorded', fontsize=16)

        plot_pos = 0
        # plot wave form
        for ch in range(orig.channels):
            Visualizer.plot_wave(fig, axs[ch + plot_pos], orig.getData(ch), orig.sample_rate, 'b-', 'Original', 0.8)
            Visualizer.plot_wave(fig, axs[ch + plot_pos], rec.getData(ch), rec.sample_rate, 'r-', 'Recorded', 0.6,
                                    f'Full Waveform Comparison - {ch} channel', 'Time (s)', 'Amplitude')
        plot_pos += orig.channels

        for ch in range(orig.channels):
            Visualizer.plot_spectrum(fig, axs[ch + plot_pos], orig.getData(ch), orig.sample_rate,
                                    f'Original Spectrogram - {ch} channel', 'Time [sec]', 'Frequency [Hz]')
        plot_pos += orig.channels

        for ch in range(rec.channels):
            Visualizer.plot_spectrum(fig, axs[ch + plot_pos], rec.getData(ch), rec.sample_rate,
                                    f'Recorded Spectrogram - {ch} channel', 'Time [sec]', 'Frequency [Hz]')
        plot_pos += rec.channels

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(path)
        #plt.show()