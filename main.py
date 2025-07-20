import argparse
from lib.AudioData import AudioData
from lib.Visualizer import Visualizer
from lib.Testcase import Testcase

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio correlation checking tool")
    parser.add_argument('--test', action='store_true', help='run test')
    parser.add_argument('--input', action='store', type=str, default='test_audio', help='dir for test audio')
    parser.add_argument('--output', action='store', type=str, default='result', help='dir for test result')
    args = parser.parse_args()

    if args.test:
        Testcase.run_test(args)