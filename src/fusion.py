"""
fusion.py : Shell and Script Execution integrationw with Core Fusion (used as endpoint in executable)
Developed by Aditya Gaur
Github : @xdityagr 
Email  : adityagaur.home@gmail.com 
"""
from fusion_core import run, Void, List 
from utils import *
import time
import sys, platform
import argparse
from colorama import Fore
from datetime import datetime


FUSION_VERSION = "0.1.0-alpha"
BRANCH = "dev"
BUILD_DATE = datetime.now().strftime("%b %d %Y") 

PYTHON_VERSION = sys.version.split()[0] 
INTERPRETER = f"Py {PYTHON_VERSION}"

ARCH = "64 bit (x86_64)" if sys.maxsize > 2**32 else "32 bit (x86)"

PLATFORM = platform.system().lower()
if PLATFORM == "windows":
    PLATFORM = "win32"

VERSION_STRING = f"Fusion {FUSION_VERSION} ({BRANCH}, {BUILD_DATE}) [{INTERPRETER}, {ARCH}] on {PLATFORM}"

def run_shell():
    """Launch the Fusion shell"""

    print(accent(f'{VERSION_STRING}', bold=True))
    
    while True:
        try:
            text = input(accent('>>> ', bold=True))
            if text.strip() == '':
                continue

            start = time.time()
            result, error = run('<stdin>', text)
            end = time.time()

            if error:
                print(accent(error.as_string(), custom_color=Fore.RED))

            elif result is not None:
                if isinstance(result, List):
                    result = result.elements[0]
                if not isinstance(result, Void):
                    print(result)

        except KeyboardInterrupt:
            print("\nExiting Fusion shell.")
            sys.exit(0)

        except Exception as e:
            print(accent(f"Fusion Shell Error: {str(e)}", custom_color=Fore.RED))

def run_file(filename):
    """Execute a Fusion script from a file (*.fs)."""

    try:
        with open(filename, 'r') as f:
            script = f.read()

        result, error = run(filename, script)
        
        if error:
            print(accent(error.as_string(), custom_color=Fore.RED))
            sys.exit(1)

    except FileNotFoundError:
        print(accent(f"File '{filename}' not found", custom_color=Fore.RED))
        sys.exit(1)

    except Exception as e:
        print(accent(f"Error loading '{filename}', {str(e)}", custom_color=Fore.RED))
        sys.exit(1)

def main():
    """Handle script execution or shell"""

    parser = argparse.ArgumentParser(description="Fusion v-Alpha0.1.0 ")

    parser.add_argument('filename', nargs='?', default=None, help="Fusion script to execute.")
    args = parser.parse_args()

    if args.filename is None:
        run_shell()
    else:
        run_file(args.filename)

if __name__ == '__main__':
    main()
