from chisquarecosmo import register_model

from . import cpl, one


def init_register_models():
    """Function for registering the built-in models."""
    register_model(cpl.model)
    register_model(one.model)
