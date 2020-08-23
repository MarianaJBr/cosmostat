import importlib
import os
import pathlib
import sys

# Environment variables that control the location of the external models.
EXTRA_PATH_ENV = "CHI2COSMO_EXTRA_PATH"
ENTRY_POINT_ENV = "CHI2COSMO_EXTRA_ENTRY_POINT"
ENTRY_POINT = "my_chisquarecosmo_models"


def plug_external_models():
    """Load the external models inside a specified directory."""
    extra_path_env = os.getenv(EXTRA_PATH_ENV)
    if extra_path_env is None:
        extra_path = pathlib.Path.cwd()
    else:
        extra_path = pathlib.Path(extra_path_env).resolve()
    # Update Pythonpath.
    sys.path.insert(0, str(extra_path))
    # We just import the module. The module and other options must be
    # registered in the module body.
    entry_point_env = os.getenv(ENTRY_POINT_ENV)
    entry_point = entry_point_env or ENTRY_POINT
    try:
        return importlib.import_module(entry_point)
    except ImportError:
        return
