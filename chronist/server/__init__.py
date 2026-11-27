from .core import Core, default_data_dir, open_data_dir

__all__ = ["Core", "default_data_dir", "open_data_dir", "create_app"]


def create_app(*args, **kwargs):
    from .app import create_app as _create
    return _create(*args, **kwargs)
