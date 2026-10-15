Collective Variable Package
===========================

[![Linux](https://github.com/craabreu/cvpack/actions/workflows/Linux.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Linux.yaml)
[![MacOS](https://github.com/craabreu/cvpack/actions/workflows/MacOS.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/MacOS.yaml)
[![Windows](https://github.com/craabreu/cvpack/actions/workflows/Windows.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Windows.yaml)
[![Linter](https://github.com/craabreu/cvpack/actions/workflows/Linter.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Linter.yaml)
[![Docs](https://github.com/craabreu/cvpack/actions/workflows/Docs.yaml/badge.svg?branch=main)](https://craabreu.github.io/cvpack/latest)
[![Coverage](https://craabreu.github.io/cvpack/development/coverage/coverage.svg)](https://craabreu.github.io/cvpack/development/coverage)

[![Conda version](https://img.shields.io/conda/v/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![PyPI version](https://img.shields.io/pypi/v/cvpack.svg)](https://pypi.org/project/cvpack)
[![Python versions](https://img.shields.io/pypi/pyversions/cvpack.svg)](https://pypi.org/project/cvpack)
[![Conda downloads](https://img.shields.io/conda/dn/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![PyPI downloads](https://img.shields.io/pypi/dm/cvpack.svg)](https://pypi.org/project/cvpack)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg)](https://github.com/craabreu/cvpack/blob/main/LICENSE)

Overview
--------

Collective variables (CVs) are functions of the coordinates of a molecular system that project its
conformational state onto a lower-dimensional space. Biasing the dynamics of a judiciously chosen
set of CVs enhances the sampling of configuration space, including regions that are otherwise
difficult to reach. The free energy of the system as a function of these CVs characterizes the
relative stability of different states and the pathways connecting them.

CVPack provides ready-to-use CVs for the molecular dynamics engine [OpenMM]. Every CV is a
subclass of OpenMM's [Force] class, so it can be added directly to a [CustomCVForce] or used to
define a [BiasVariable] for [Metadynamics]. In addition, each CV:

- reports its value and its effective mass in OpenMM's MD unit system;
- can be serialized to and deserialized from YAML;
- can be reported during a simulation through `cvpack.reporting.StateDataReporter`.

Collective Variables
--------------------

| Collective Variable          | Description                                                               |
|------------------------------|---------------------------------------------------------------------------|
| [Angle]                      | angle formed by three atoms                                               |
| [Atomic function]            | user-defined function of the coordinates of groups of atoms               |
| [Attraction strength]        | strength of the attraction between two groups of atoms                    |
| [Centroid function]          | user-defined function of the centroids of groups of atoms                 |
| [Composite RMSD]             | multibody RMSD with concerted-rotation alignment                          |
| [Distance]                   | distance between two atoms                                                |
| [Helix angle content]        | alpha-helix angle content of a sequence of residues                       |
| [Helix H-bond content]       | alpha-helix hydrogen-bond content of a sequence of residues               |
| [Helix RMSD content]         | alpha-helix RMSD content of a sequence of residues                        |
| [Helix torsion content]      | alpha-helix Ramachandran content of a sequence of residues                |
| [Meta-collective variable]   | function of other collective variables                                    |
| [Number of contacts]         | number of contacts between two groups of atoms                            |
| [OpenMM force wrapper]       | potential energy of an OpenMM Force object, used as a CV                  |
| [Path in CV space]           | progress along (or deviation from) a path in CV space                     |
| [Path in RMSD space]         | progress along (or deviation from) a path in RMSD space                   |
| [Radius of gyration]         | radius of gyration of a group of atoms                                    |
| [Radius of gyration squared] | square of the radius of gyration of a group of atoms                      |
| [Residue coordination]       | number of contacts between two disjoint groups of residues                |
| [RMSD]                       | root-mean-square deviation with respect to a reference structure          |
| [Sheet RMSD content]         | beta-sheet RMSD content of a sequence of residues                         |
| [Shortest distance]          | smooth approximation of the shortest distance between two groups of atoms |
| [Torsion]                    | torsion angle formed by four atoms                                        |
| [Torsion similarity]         | degree of similarity between pairs of torsion angles                      |

Installation
------------

CVPack is available from the [mdtools](https://anaconda.org/mdtools/cvpack) conda channel and from
[PyPI](https://pypi.org/project/cvpack):

```bash
conda install -c conda-forge -c mdtools cvpack
```

```bash
pip install cvpack
```

`CompositeRMSD` also requires the [openmm-cpp-forces] package, which is available from the
mdtools channel for OpenMM 8.1.

Quick Start
-----------

The example below computes the backbone dihedral angles of alanine dipeptide, using a test system
from [OpenMMTools]:

```python
import cvpack
import openmm
from openmmtools import testsystems

model = testsystems.AlanineDipeptideVacuum()
phi = cvpack.Torsion(4, 6, 8, 14, name="phi")
psi = cvpack.Torsion(6, 8, 14, 16, name="psi")
for cv in (phi, psi):
    cv.addToSystem(model.system)

context = openmm.Context(model.system, openmm.VerletIntegrator(0.001))
context.setPositions(model.positions)
print(phi.getValue(context), psi.getValue(context))
```

`addToSystem` puts each CV in its own force group, which makes evaluating it cheap. Note that a
CV added to a system also contributes its value to the potential energy. To compute CVs without
biasing the dynamics, integrate only the force groups of the physical forces (see
`openmm.Integrator.setIntegrationForceGroups`).

Documentation
-------------

The documentation of the [latest release](https://craabreu.github.io/cvpack/latest) and of the
[development version](https://craabreu.github.io/cvpack/development) is available on GitHub Pages.

Related Packages
----------------

[OpenMM-Colvars] adds the [Colvars] library to OpenMM through a C++ `Force` with Python bindings.
Colvars has its own set of CVs and enhanced-sampling methods, and it is also available in NAMD,
LAMMPS, GROMACS, VMD and Tinker-HP.

Copyright
---------

Copyright (c) 2023-2024 [C. Abreu](https://github.com/craabreu) & Redesign Science

Acknowledgments
---------------

Initial project based on the [CMS Cookiecutter] version 1.1.

[BiasVariable]:      https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
[CMS Cookiecutter]:  https://github.com/molssi/cookiecutter-cms
[Colvars]:           https://github.com/Colvars/colvars
[CustomCVForce]:     https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:             https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[Metadynamics]:      https://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
[OpenMM]:            https://openmm.org
[OpenMM-Colvars]:    https://github.com/fhh2626/OpenMM-Colvars
[OpenMMTools]:       https://openmmtools.readthedocs.io
[openmm-cpp-forces]: https://anaconda.org/mdtools/openmm-cpp-forces

[Angle]:                      https://craabreu.github.io/cvpack/latest/api/Angle.html
[Atomic function]:            https://craabreu.github.io/cvpack/latest/api/AtomicFunction.html
[Attraction strength]:        https://craabreu.github.io/cvpack/latest/api/AttractionStrength.html
[Centroid function]:          https://craabreu.github.io/cvpack/latest/api/CentroidFunction.html
[Composite RMSD]:             https://craabreu.github.io/cvpack/latest/api/CompositeRMSD.html
[Distance]:                   https://craabreu.github.io/cvpack/latest/api/Distance.html
[Helix angle content]:        https://craabreu.github.io/cvpack/latest/api/HelixAngleContent.html
[Helix H-bond content]:       https://craabreu.github.io/cvpack/latest/api/HelixHBondContent.html
[Helix RMSD content]:         https://craabreu.github.io/cvpack/latest/api/HelixRMSDContent.html
[Helix torsion content]:      https://craabreu.github.io/cvpack/latest/api/HelixTorsionContent.html
[Meta-collective variable]:   https://craabreu.github.io/cvpack/latest/api/MetaCollectiveVariable.html
[Number of contacts]:         https://craabreu.github.io/cvpack/latest/api/NumberOfContacts.html
[OpenMM force wrapper]:       https://craabreu.github.io/cvpack/latest/api/OpenMMForceWrapper.html
[Path in CV space]:           https://craabreu.github.io/cvpack/latest/api/PathInCVSpace.html
[Path in RMSD space]:         https://craabreu.github.io/cvpack/latest/api/PathInRMSDSpace.html
[Radius of gyration]:         https://craabreu.github.io/cvpack/latest/api/RadiusOfGyration.html
[Radius of gyration squared]: https://craabreu.github.io/cvpack/latest/api/RadiusOfGyrationSq.html
[Residue coordination]:       https://craabreu.github.io/cvpack/latest/api/ResidueCoordination.html
[RMSD]:                       https://craabreu.github.io/cvpack/latest/api/RMSD.html
[Sheet RMSD content]:         https://craabreu.github.io/cvpack/latest/api/SheetRMSDContent.html
[Shortest distance]:          https://craabreu.github.io/cvpack/latest/api/ShortestDistance.html
[Torsion]:                    https://craabreu.github.io/cvpack/latest/api/Torsion.html
[Torsion similarity]:         https://craabreu.github.io/cvpack/latest/api/TorsionSimilarity.html
