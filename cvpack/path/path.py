"""
.. module:: path
   :platform: Linux, MacOS, Windows
   :synopsis: Specifications for Path Collective Variables

.. classauthor:: Charlles Abreu <craabreu@gmail.com>

"""

from ..serialization import Serializable


class Metric(Serializable):
    """
    A measure of progress or deviation with respect to a path in CV space
    """

    yaml_tag = "!cvpack.path.Metric"

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Metric) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __getstate__(self) -> dict:
        return {"name": self.name}

    def __setstate__(self, state: dict) -> None:
        self.name = state["name"]


Metric.registerTag("!cvpack.path.Metric")


progress: Metric = Metric("progress")
deviation: Metric = Metric("deviation")
