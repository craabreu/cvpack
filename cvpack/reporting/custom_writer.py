"""
.. class:: CustomWriter
   :platform: Linux, MacOS, Windows
   :synopsis: An abstract class for StateDataReporter writers

.. moduleauthor:: Charlles Abreu <craabreu@gmail.com>

"""

import typing as t

from openmm import app as mmapp


@t.runtime_checkable
class CustomWriter(t.Protocol):
    """
    An abstract class for StateDataReporter writers
    """

    def initialize(self, simulation: mmapp.Simulation) -> None:
        """
        Initializes the writer. This method is called before the first report and
        can be used to perform any necessary setup.

        Parameters
        ----------
        simulation
            The simulation object.
        """

    def getHeaders(self) -> list[str]:
        """
        Gets a list of strigs containing the headers to be added to the report.
        """
        raise NotImplementedError("Method 'getHeaders' not implemented")

    def getValues(self, simulation: mmapp.Simulation) -> list[float]:
        """
        Gets a list of floats containing the values to be added to the report.

        Parameters
        ----------
        simulation
            The simulation object.
        """
        raise NotImplementedError("Method 'getValues' not implemented")
