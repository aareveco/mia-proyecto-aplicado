from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class PubChemService(ABC):
    """
    Interface for interacting with PubChem database.
    """
    @abstractmethod
    def get_compound_by_mz(self, mz: float, tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Search for a compound by mass-to-charge ratio (m/z).
        Returns a dictionary with compound details or None if not found.
        """
        pass
