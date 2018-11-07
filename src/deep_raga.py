"""
file: deep_raga.py
author: Afnan Enayet

This module serves as the entry point to the function. It contains the code 
necessary to parse command line arguments, and the logic that will call 
other routines from other modules to do whatever is necessary for the program. 
"""

import argparse
import sys


def main() -> int:
    """
    A wrapper for the main function. It parses command line options and calls
    the appropriate corresponding methods to execute the program.
    """
    parser = argparse.ArgumentParser(
        prog="Deep Raga",
        description="A program "
        + "that trains a neural network on MIDI "
        + "audio files, and uses that trained "
        + "network to optionally generate new "
        + "music that tries to emulate the style"
        + " of the training data.",
    )
    args = parser.parse_args()
    return 0


if __name__ == "__main__":
    sys.exit(main())
