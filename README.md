Collective Variable Package
===========================

[//]: # (Badges)
[![Linux](https://github.com/craabreu/cvpack/actions/workflows/Linux.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Linux.yaml)
[![MacOS](https://github.com/craabreu/cvpack/actions/workflows/MacOS.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/MacOS.yaml)
[![Windows](https://github.com/craabreu/cvpack/actions/workflows/Windows.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Windows.yaml)
[![Linter](https://github.com/craabreu/cvpack/actions/workflows/Linter.yaml/badge.svg?branch=main)](https://github.com/craabreu/cvpack/actions/workflows/Linter.yaml)
[![Docs](https://github.com/craabreu/cvpack/actions/workflows/Docs.yaml/badge.svg?branch=main)](https://craabreu.github.io/cvpack/development)
[![Coverage Report](https://craabreu.github.io/cvpack/development/coverage/coverage.svg)](https://craabreu.github.io/cvpack/development/coverage)

[![Conda version](https://img.shields.io/conda/v/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![Conda platforms](https://img.shields.io/conda/pn/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![Conda downloads](https://img.shields.io/conda/dn/mdtools/cvpack.svg)](https://anaconda.org/mdtools/cvpack)
[![PyPI version](https://img.shields.io/pypi/v/cvpack.svg)](https://pypi.org/project/cvpack)
[![PyPI version](https://img.shields.io/pypi/pyversions/cvpack.svg)](https://pypi.org/project/cvpack)
[![PyPI version](https://img.shields.io/pypi/dm/cvpack.svg)](https://pypi.org/project/cvpack)

[![License](https://img.shields.io/badge/License-MIT-yellowgreen.svg?style=flat)](https://github.com/craabreu/cvpack/blob/main/LICENSE)
[![Twitter](https://badgen.net/badge/About/RedesignScience)](https://www.redesignscience.com)

Overview
--------

Collective variables (CVs) are functions of the coordinates of a molecular system and provide a
means to project its conformational state onto a lower-dimensional space. By stimulating the
dynamics of a judiciously chosen set of CVs, one can obtain an enhanced sampling of the
configuration space, including regions that are otherwise difficult to access. The system's
free energy as a function of these CVs can be used to characterize the relative stability of
different states and to identify pathways connecting them.

CVPack is a Python package that provides pre-defined CVs for the powerful molecular dynamics engine
[OpenMM]. All these CVs are subclasses of OpenMM's [Force] class and, as such, can be directly added
to a [CustomCVForce] or used to define a [BiasVariable] for [Metadynamics].

Collective Variables
--------------------

The CVs implemented in CVPack are listed in the table below.

| Collective Variable     | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| [Angle]                 | angle formed by three atoms                                      |
| [Atomic Function]       | a user-defined function of the coordinates of a group of atoms   |
| [Attraction Strength]   | strength of the attraction between two groups of atoms           |
| [Centroid Function]     | a user-defined function of the centroids of groups of atoms      |
| [Composite RMSD]        | multibody RMSD with concerted-rotation alignment                 |
| [Distance]              | distance between two atoms                                       |
| [Helix angle content]   | alpha-helix angle content of a sequence of residues              |
| [Helix H-bond content]  | alpha-helix hydrogen-bond content of a sequence of residues      |
| [Helix RMSD content]    | alpha-helix RMSD content of a sequence of residues               |
| [Helix torsion content] | alpha-helix Ramachandran content of a sequence of residues       |
| [Meta CV]               | a function of other collective variables                         |
| [Number of contacts]    | number of contacts between two groups of atoms                   |
| [OpenMM Force wrapper]  | converts an OpenMM Force object into a CVPack CV                 |
| [Path in CV space]      | progress along (or deviation from) a path in CV space            |
| [Path in RMSD space]    | progress along (or deviation from) a path in RMSD space          |
| [Radius of gyration]    | radius of gyration of a group of atoms                           |
| [(Radius of gyration)^2]| square of the radius of gyration of a group of atoms             |
| [Residue coordination]  | number of contacts between two disjoint groups of residues       |
| [RMSD]                  | root-mean-square deviation with respect to a reference structure |
| [Sheet RMSD content]    | beta-sheet RMSD content of a sequence of residues                |
| [Shortest Distance]     | shortest distance between two groups of atoms                    |
| [Torsion]               | torsion angle formed by four atoms                               |
| [Torsion similarity]    | degree of similarity between pairs of torsion angles             |

Installation and Usage
----------------------

CVPack is available as a conda package on the
[mdtools](https://anaconda.org/mdtools/cvpack) channel. To install it, run:

```bash
    conda install -c conda-forge -c mdtools cvpack
```

Or:

```bash
    mamba install -c mdtools cvpack
```

To use CVPack in your own Python script or Jupyter notebook, simply import it as follows:

```python
    import cvpack
```

Documentation
-------------

Documentation for the latest CVPack version is available on [Github Pages](https://craabreu.github.io/cvpack/latest).

Copyright
---------


Copyright (c) 2023-2024 [C. Abreu](https://github.com/craabreu) & [Redesign Science](https://www.redesignscience.com)


Acknowledgments
---------------

Initial project based on the [CMS Cookiecutter] version 1.1.

[BiasVariable]:       http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.BiasVariable.html
[CMS Cookiecutter]:   https://github.com/molssi/cookiecutter-cms
[CustomCVForce]:      http://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html
[Force]:              http://docs.openmm.org/latest/api-python/generated/openmm.openmm.Force.html
[Metadynamics]:       http://docs.openmm.org/latest/api-python/generated/openmm.app.metadynamics.Metadynamics.html
[OpenMM]:             http://openmm.org

[Angle]:                  https://craabreu.github.io/cvpack/latest/api/Angle.html
[Atomic Function]:        https://craabreu.github.io/cvpack/latest/api/AtomicFunction.html
[Attraction Strength]:    https://craabreu.github.io/cvpack/latest/api/AttractionStrength.html
[Centroid Function]:      https://craabreu.github.io/cvpack/latest/api/CentroidFunction.html
[Composite RMSD]:         https://craabreu.github.io/cvpack/latest/api/CompositeRMSD.html
[Distance]:               https://craabreu.github.io/cvpack/latest/api/Distance.html
[Helix angle content]:    https://craabreu.github.io/cvpack/latest/api/HelixAngleContent.html
[Helix H-bond content]:   https://craabreu.github.io/cvpack/latest/api/HelixHBondContent.html
[Helix RMSD content]:     https://craabreu.github.io/cvpack/latest/api/HelixRMSDContent.html
[Helix torsion content]:  https://craabreu.github.io/cvpack/latest/api/HelixTorsionContent.html
[Meta CV]:                https://craabreu.github.io/cvpack/latest/api/MetaCollectiveVariable.html
[Number of contacts]:     https://craabreu.github.io/cvpack/latest/api/NumberOfContacts.html
[OpenMM Force wrapper]:   https://craabreu.github.io/cvpack/latest/api/OpenMMForceWrapper.html
[Path in CV space]:       https://craabreu.github.io/cvpack/latest/api/PathInCVSpace.html
[Path in RMSD space]:     https://craabreu.github.io/cvpack/latest/api/PathInRMSDSpace.html
[Radius of gyration]:     https://craabreu.github.io/cvpack/latest/api/RadiusOfGyration.html
[(Radius of gyration)^2]: https://craabreu.github.io/cvpack/latest/api/RadiusOfGyrationSq.html
[Residue coordination]:   https://craabreu.github.io/cvpack/latest/api/ResidueCoordination.html
[RMSD]:                   https://craabreu.github.io/cvpack/latest/api/RMSD.html
[Sheet RMSD content]:     https://craabreu.github.io/cvpack/latest/api/SheetRMSDContent.html
[Shortest Distance]:      https://craabreu.github.io/cvpack/latest/api/ShortestDistance.html
[Torsion]:                https://craabreu.github.io/cvpack/latest/api/Torsion.html
[Torsion similarity]:     https://craabreu.github.io/cvpack/latest/api/TorsionSimilarity.html
