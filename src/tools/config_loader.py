from pathlib import Path
from typing import Any

import yaml


class Config:
    """A class to manage configuration settings.

    Attributes
    ----------
    cfg_dict : dict
        The dictionary containing the configuration settings.

    """

    def __init__(self, cfg_dict: dict):
        self._update_cfg(cfg_dict)

    def _update_cfg(self, cfg_dict: dict):
        """Update the configuration settings recursively.

        Parameters
        ----------
        cfg_dict : dict
            The dictionary containing the configuration settings.

        """
        if not isinstance(cfg_dict, dict):
            try:
                cfg_dict = cfg_dict.__dict__
            except AttributeError as err:
                raise ValueError(f"Passed a config that does not have a dictionary ({type(cfg_dict)})") from err

        for key, val in cfg_dict.items():
            if isinstance(val, dict):
                val = Config(val)

            if isinstance(key, int):
                key = f"{key}"

            setattr(self, key, val)

    def __getitem__(self, key: str):
        return self.get(key)

    def get(self, key: str, default: Any | None = None) -> Any:
        return getattr(self, key, default)

    @property
    def dict(self):
        """Return dict from attributes."""
        return {key: getattr(self, key) for key in vars(self) if not key.startswith("_")}

    def __repr__(self) -> str:
        """Very Magic method."""
        config_params = vars(self)

        return str(config_params)


def load_config(cfg_path: str | Path) -> Config:
    """Function for load config file.

    Parameters
    ----------
    cfg_path: str | Path
        path to config file

    Returns
    -------
    Config
        Full config as class with console arguments

    """
    try:
        with open(cfg_path) as ymfile:
            cfg = yaml.load(ymfile, Loader=yaml.FullLoader)
    except:
        raise Exception(f"Файл {cfg_path} не найден!")


    cfg = Config(cfg)

    return cfg
