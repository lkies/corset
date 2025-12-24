from dataclasses import fields
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import yaml


class YamlSerializableMixin:
    """Mixin class to add YAML serialization/deserialization to dataclasses and a few other common types."""

    class _Dumper(yaml.SafeDumper):
        pass

    class _Loader(yaml.SafeLoader):
        pass

    @classmethod
    def _register_misc_classes(cls) -> None:
        Dumper = cls._Dumper  # noqa: N806
        Loader = cls._Loader  # noqa: N806

        # python built-in classes
        @staticmethod
        def complex_representer(dumper: yaml.Dumper, data: complex) -> yaml.Node:
            return dumper.represent_sequence("!complex", [data.real, data.imag], flow_style=True)

        def complex_constructor(loader: yaml.Loader, node: yaml.Node) -> complex:
            return complex(*loader.construct_sequence(node, deep=True))  # pyright: ignore[reportArgumentType]

        Dumper.add_representer(complex, complex_representer)
        Loader.add_constructor("!complex", complex_constructor)  # pyright: ignore[reportArgumentType]

        def tuple_representer(dumper: yaml.Dumper, data: tuple) -> yaml.Node:
            return dumper.represent_sequence("!tuple", list(data))

        def tuple_constructor(loader: yaml.Loader, node: yaml.Node) -> tuple:
            return tuple(loader.construct_sequence(node, deep=True))  # pyright: ignore[reportArgumentType]

        Dumper.add_representer(tuple, tuple_representer)  # pyright: ignore[reportArgumentType]
        Loader.add_constructor("!tuple", tuple_constructor)  # pyright: ignore[reportArgumentType]

        # numpy classes
        def ndarray_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
            return dumper.represent_sequence("!ndarray", data.tolist())

        def ndarray_constructor(loader: yaml.Loader, node: yaml.Node) -> np.ndarray:
            return np.array(loader.construct_sequence(node, deep=True))  # pyright: ignore[reportArgumentType]

        Dumper.add_representer(np.ndarray, ndarray_representer)  # pyright: ignore[reportArgumentType]
        Loader.add_constructor("!ndarray", ndarray_constructor)  # pyright: ignore[reportArgumentType]

        Dumper.add_representer(np.float32, lambda dumper, data: dumper.represent_float(float(data)))
        Dumper.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))
        Dumper.add_representer(np.complex64, complex_representer)  # pyright: ignore[reportArgumentType]
        Dumper.add_representer(np.complex128, complex_representer)

    @staticmethod
    def __representer(dumper: yaml.Dumper, data: Any):
        # only store the dataclass fields
        attrs = {field.name: getattr(data, field.name) for field in fields(data)}
        return dumper.represent_mapping(f"!{type(data).__name__}", attrs)

    @classmethod
    def __constructor(cls, loader: yaml.Loader, node: yaml.Node):
        return cls(**loader.construct_mapping(node, deep=True))  # pyright: ignore[reportCallIssue, reportArgumentType]

    def __init_subclass__(cls) -> None:
        # access via base class not dynamic subclass
        YamlSerializableMixin._Dumper.add_representer(cls, cls.__representer)  # pyright: ignore[reportArgumentType]
        YamlSerializableMixin._Loader.add_constructor(f"!{cls.__name__}", cls.__constructor)

    def save_yaml(self: Any, filename: str | Path) -> None:
        """Save the object to a YAML file.

        Args:
            filename: Path to the YAML file.
        """
        from . import __version__

        yaml_data = {
            "meta": {"corset_version": __version__, "timestamp": pd.Timestamp.now().isoformat()},
            "data": self,
        }
        filename = Path(filename)
        filename.write_text(yaml.dump(yaml_data, Dumper=YamlSerializableMixin._Dumper, sort_keys=False))

    @classmethod
    def load_yaml(cls, filename: str | Path) -> Self:
        """Load an object from a YAML file.

        If this is called from a subclass the loaded object type must match that subclass.

        Args:
            filename: Path to the YAML file.

        Returns:
            The loaded object.

        Raises:
            ValueError: If the YAML file does not contain a 'data' field.
            TypeError: If the loaded object type does not match the class used to call this method.
        """

        filename = Path(filename)
        yaml_data = yaml.load(filename.read_text(), Loader=YamlSerializableMixin._Loader)  # noqa: S506
        if "data" not in yaml_data:
            raise ValueError("YAML file does not contain 'data' field.")
        data = yaml_data["data"]
        if cls is not YamlSerializableMixin and not isinstance(data, cls):
            raise TypeError(
                f"Attempting to load object of type '{type(data).__name__}' through different type '{cls.__name__}'."
            )
        return data


YamlSerializableMixin._register_misc_classes()

save_yaml = YamlSerializableMixin.save_yaml
"""Save any :class:`YamlSerializableMixin` instance or basic collection containing
them to a YAML file. This is an alias for :meth:`YamlSerializableMixin.save_yaml`.
"""
load_yaml = YamlSerializableMixin.load_yaml
"""Load any :class:`YamlSerializableMixin` instance or basic collection containing
them from a YAML file. This is an alias for :meth:`YamlSerializableMixin.load_yaml`."""
