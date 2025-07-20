import os
from lib.AudioData import AudioData
from lib.Visualizer import Visualizer
from lib.AudioFilter import AudioFilter, AudioNoise
from lib.AudioEvaluator import AudioEvaluator

class Testcase:
    @staticmethod
    def create_audio_test(target_dir):
        # mkdirs target_dir
        os.makedirs(target_dir, exist_ok=True)
        # 1. Make sine
        input_audio = AudioData.from_sine(
            duration=3,
            freq=10,
            rate=48000,
            width=4,
            channels=2,
        )

        # 2. Copy from input
        output_audio = input_audio.copy()

        # 3. Add noise
        output_audio.apply(AudioNoise.clipping, multiple=0.8) \
                    .apply(AudioNoise.dc_offset, offset=output_audio.max_amp * 0.2) \
                    .apply(AudioNoise.pop_noise, target_sec=1.0) \
                    .apply(AudioNoise.pop_noise, target_sec=2.2, target_channel=[1],
                           noise_level=output_audio.max_amp, noise_duration_samples=10)

        # # add normalized noise with 2% of max value
        # #output_audio = input_audio.copy()
        output_audio.apply(AudioNoise.normalized_noise, noise_level=output_audio.max_amp * 0.02)

        # # add normalized noise with 20% of max data
        # output_audio = input_audio.copy()
        # output_audio.apply(AudioNoise.normalized_noise, noise_level=output_audio.max_val * 0.2)

        # # audio cut noise
        # output_audio = input_audio.copy()
        # output_audio.apply(AudioNoise.cut_noise, target_sec=1.0, target_channel=[0], noise_duration_samples=50)

        # # audio filter
        # input_audio= AudioData.from_multi_sine(
        #     duration=3,
        #     freqs=[100, 2000],
        #     rate=48000,
        #     width=4,
        #     channels=2
        # )
        # output_audio = input_audio.copy()
        # output_audio.apply(AudioFilter.freq_pass_filter, filter_type='low', cutoff_freq=500)

        # output_audio = input_audio.copy()
        # output_audio.apply(AudioFilter.freq_pass_filter, filter_type='high', cutoff_freq=500)

        # output_audio = input_audio.copy()
        # output_audio.apply(AudioFilter.band_pass_filter, low_cutoff=100, high_cutoff=500)

        # output_audio = input_audio.copy()
        # output_audio.apply(AudioFilter.band_pass_filter, low_cutoff=500, high_cutoff=2000)

        # output_audio = input_audio.copy()
        # output_audio.apply(AudioFilter.band_stop_filter, low_cutoff=50, high_cutoff=1000)

        input_audio.save(f"{target_dir}/input.wav")
        output_audio.save(f"{target_dir}/output.wav")

    @staticmethod
    def visualize_audio_test(input_dir, output_dir):
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        input_audio = AudioData.from_file(f"{input_dir}/input.wav")
        output_audio = AudioData.from_file(f"{input_dir}/output.wav")
        path = f"{output_dir}/visualized_audio.png"
        Visualizer.plot_audio_data(path, input_audio, output_audio)

    @staticmethod
    def evaluate_audio_test(input_dir):
        """
        evaluate audio test
        """
        input_audio = AudioData.from_file(f"{input_dir}/input.wav")
        output_audio = AudioData.from_file(f"{input_dir}/output.wav")
        
        evaluation_results = AudioEvaluator.evaluate(input_audio, output_audio)

        # 보기 쉽게 결과 출력
        print("\n\n================================")
        print("    Correlation Evaluation Results")
        print("================================")
        for ch, metrics in evaluation_results.items():
            print(f"\n▶ Channel {ch.split('_')[-1]}")
            for key, value in metrics.items():
                # Show key in a more readable format
                key_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    print(f"  - {key_name}: {value:.4f}")
                else:
                    print(f"  - {key_name}: {value}")
        print("\n================================")

    @staticmethod
    def run_test(args):
        print(args)
        Testcase.create_audio_test(args.input)
        Testcase.visualize_audio_test(args.input, args.output)
        Testcase.evaluate_audio_test(args.input)