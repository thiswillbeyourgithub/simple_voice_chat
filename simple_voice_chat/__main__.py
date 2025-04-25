"""
Entry point for running the Simple Voice Chat application as a module
using `python -m simple_voice_chat`.
"""

import sys
from .simple_voice_chat import main

if __name__ == "__main__":
    # Optionally, you could handle command-line arguments specifically for
    # module execution here, but typically it's cleaner to let the main()
    # function handle argparse as it already does.
    sys.exit(main())
