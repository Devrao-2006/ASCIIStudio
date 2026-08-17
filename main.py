import argparse
import sys
import os

# Ensure app root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import run_server
from converters.image_converter import convert_image
from converters.video_converter import convert_video

def main():
    """
    Main entry point for ASCII Studio & CLI Converters.
    """
    parser = argparse.ArgumentParser(
        description="ASCII Studio — GPU-Accelerated Real-Time ASCII Engine & Studio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                      Launch full ASCII Studio Web App (Webcam, Photos, Videos)
  python main.py image in.png out.png Convert a single photo via CLI
  python main.py video in.mp4 out.mp4 Convert a video file via CLI
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run (defaults to Studio App if omitted).")

    # Studio / App / Webcam Subcommand
    studio_parser = subparsers.add_parser("studio", aliases=["app", "webcam"], help="Launch ASCII Studio Web App.")
    studio_parser.add_argument("--device", "-d", type=int, default=None, help="Camera device index. Auto-detects if omitted.")

    # Image Subcommand
    image_parser = subparsers.add_parser("image", help="Convert a single image to ASCII via CLI.")
    image_parser.add_argument("input", type=str, help="Path to input image file.")
    image_parser.add_argument("output", type=str, help="Path to save output image file.")

    # Video Subcommand
    video_parser = subparsers.add_parser("video", help="Convert a video file to ASCII via CLI.")
    video_parser.add_argument("input", type=str, help="Path to input video file.")
    video_parser.add_argument("output", type=str, help="Path to save output video file (.mp4 recommended).")

    args = parser.parse_args()

    # Default to launching ASCII Studio if no subcommand provided
    if args.mode is None or args.mode in ["studio", "app", "webcam"]:
        device_idx = getattr(args, "device", None)
        run_server(device_index=device_idx)
    elif args.mode == "image":
        convert_image(args.input, args.output)
    elif args.mode == "video":
        convert_video(args.input, args.output)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit, BaseException):
        sys.exit(0)