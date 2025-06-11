import os
import sys
from typing import Any, Dict

import pytest

# Add tests folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_utils import SynthTextDirectory

# Store the fixture value in a dict
_synth_text_dirs: Dict[str, Any] = {}


@pytest.fixture(scope="session")
def synth_text_dir(request):
    """Create a shared SynthTextDirectory once for the entire test session."""
    # Get the worker id
    worker_id = request.config.getoption("--gw", default="master")

    # If this is the master process or first process
    if worker_id == "master" or not _synth_text_dirs:
        # Create the directory
        with SynthTextDirectory() as tmp_datadir:
            # Store for all workers to use
            _synth_text_dirs["directory"] = tmp_datadir
            yield tmp_datadir
    else:
        # Return existing directory
        yield _synth_text_dirs["directory"]
