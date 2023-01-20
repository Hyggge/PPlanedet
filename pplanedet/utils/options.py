import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PPLanedet')
    parser.add_argument(
        '-c', '--config-file', metavar="FILE", help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    # cuda setting
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    # checkpoint and log
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='put the path to resuming file if needed')
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='put the path to resuming file if needed')
    # for evaluation
    parser.add_argument(
        '--evaluate-only',
        action='store_true',
        default=False,
        help='skip validation during training')
    # config options
    parser.add_argument(
        'opts',
        help='See config for all options',
        default=None,
        nargs=argparse.REMAINDER)

    #for inference
    parser.add_argument(
        "--source_path",
        default="",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_dir", default="", help="path to reference images")
    parser.add_argument("--model_path", default=None, help="model for loading")

    args = parser.parse_args()

    return args