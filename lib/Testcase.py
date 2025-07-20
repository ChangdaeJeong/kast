import os
from lib.AudioData import AudioData
from lib.Visualizer import Visualizer

class Testcase:
    @staticmethod
    def create_audio_test(target_dir):
        # mkdirs target_dir
        os.makedirs(target_dir, exist_ok=True)
        # 1. Make sine
        input_audio = AudioData.from_sine(
            duration=3,
            freq=100,
            rate=48000,
            width=4,
            channels=2,
        )
        input_audio.save(f"{target_dir}/input.wav")

        # 2. Copy from input
        output_audio = input_audio.copy()
        
        # 3. Add noise
        output_audio.add_dc_offset(offset_level=2000) \
                    .add_pop_noise(target_sec=1.0) \
                    #.add_clipping(gain=1.8) \

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
    def run_test(args):
        print(args)
        Testcase.create_audio_test(args.input)
        Testcase.visualize_audio_test(args.input, args.output)