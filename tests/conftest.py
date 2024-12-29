# thiis is a special file recognized by pytest
# define fixtures or perform path manipulations here to make them accessible across all test files.

import sys
import pytest
from pathlib import Path

# Add the project root to sys.path

#The goal is to ensure that Python can find and import modules from the project's root directory (where the main code resides), even if the script is being executed from a subdirectory (like a tests folder).
# .resolve gives the full absolute path
# .parent.parent gives the directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# index 0 means that the project root will be the first place Python looks for modules to import
sys.path.insert(0, str(PROJECT_ROOT))

# Define fixtures 
# defines a fixture in the pytest testing framework. A fixture is a reusable piece of code that can be used in multiple test functions to provide setup or shared data needed for testing.
# The scope='session' argument specifies that this fixture will be created once per test session, meaning it’s shared across all test functions in the test run.
@pytest.fixture(scope='session')
def sample_config():
    return {
        # Return a sample configuration dictionary
    }
