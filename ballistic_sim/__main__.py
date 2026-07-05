"""Allow running the package as ``python -m ballistic_sim``."""

import sys
from ballistic_sim.cli import main

if __name__ == "__main__":
    main()
    sys.exit(0)
