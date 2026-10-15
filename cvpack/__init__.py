"""
.. package:: cvpack
    :platform: Linux, MacOS, Windows
    :synopsis: Collective Variables for Molecular Dynamics Simulations with OpenMM
"""

try:
    from ._version import __version__  # noqa: F401
except ImportError:  # source tree that was never built or installed
    __version__ = "0+unknown"

from .angle import Angle  # noqa: F401
from .atomic_function import AtomicFunction  # noqa: F401
from .attraction_strength import AttractionStrength  # noqa: F401
from .centroid_function import CentroidFunction  # noqa: F401
from .collective_variable import CollectiveVariable  # noqa: F401
from .composite_rmsd import CompositeRMSD  # noqa: F401
from .distance import Distance  # noqa: F401
from .helix_angle_content import HelixAngleContent  # noqa: F401
from .helix_hbond_content import HelixHBondContent  # noqa: F401
from .helix_rmsd_content import HelixRMSDContent  # noqa: F401
from .helix_torsion_content import HelixTorsionContent  # noqa: F401
from .meta_collective_variable import MetaCollectiveVariable  # noqa: F401
from .number_of_contacts import NumberOfContacts  # noqa: F401
from .openmm_force_wrapper import OpenMMForceWrapper  # noqa: F401
from .path_in_cv_space import PathInCVSpace  # noqa: F401
from .path_in_rmsd_space import PathInRMSDSpace  # noqa: F401
from .radius_of_gyration import RadiusOfGyration  # noqa: F401
from .radius_of_gyration_sq import RadiusOfGyrationSq  # noqa: F401
from .residue_coordination import ResidueCoordination  # noqa: F401
from .rmsd import RMSD  # noqa: F401
from .sheet_rmsd_content import SheetRMSDContent  # noqa: F401
from .shortest_distance import ShortestDistance  # noqa: F401
from .torsion import Torsion  # noqa: F401
from .torsion_similarity import TorsionSimilarity  # noqa: F401

__all__ = [
    "Angle",
    "AtomicFunction",
    "AttractionStrength",
    "CentroidFunction",
    "CollectiveVariable",
    "CompositeRMSD",
    "Distance",
    "HelixAngleContent",
    "HelixHBondContent",
    "HelixRMSDContent",
    "HelixTorsionContent",
    "MetaCollectiveVariable",
    "NumberOfContacts",
    "OpenMMForceWrapper",
    "PathInCVSpace",
    "PathInRMSDSpace",
    "RadiusOfGyration",
    "RadiusOfGyrationSq",
    "ResidueCoordination",
    "RMSD",
    "SheetRMSDContent",
    "ShortestDistance",
    "Torsion",
    "TorsionSimilarity",
]
