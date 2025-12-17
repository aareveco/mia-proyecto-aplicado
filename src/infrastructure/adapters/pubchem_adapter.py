import requests
import time
from typing import Optional, Dict, Any, List
from src.application.ports.pubchem_port import PubChemService

class PubChemAdapter(PubChemService):
    """
    Adapter for PubChem PUG REST API.
    """
    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def get_compound_by_mz(self, mz: float, tolerance: float = 0.01) -> Optional[Dict[str, Any]]:
        """
        Search for compounds within mz +/- tolerance range.
        Returns details of the first few matches.
        """
        min_mz = mz - tolerance
        max_mz = mz + tolerance
        
        # 1. Search CIDs by Mass Range
        # URL: .../compound/monoisotopic_mass/range/{min}/{max}/cids/JSON
        search_url = f"{self.BASE_URL}/compound/monoisotopic_mass/range/{min_mz}/{max_mz}/cids/JSON"
        
        try:
            response = requests.get(search_url, timeout=5)
            if response.status_code != 200:
                print(f"[PubChem] No results or error for mass range {min_mz}-{max_mz}. Status: {response.status_code}")
                return None
            
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            
            if not cids:
                return None
            
            # Limit to top 5 to avoid huge payloads
            top_cids = cids[:5]
            
            # 2. Fetch Details (Title/Synonyms usually most useful)
            # URL: .../compound/cid/{cids_str}/property/Title,MolecularFormula/JSON
            cids_str = ",".join(map(str, top_cids))
            details_url = f"{self.BASE_URL}/compound/cid/{cids_str}/property/Title,MolecularFormula/JSON"
            
            details_resp = requests.get(details_url, timeout=5)
            if details_resp.status_code != 200:
                print(f"[PubChem] Error fetching details for CIDs {cids_str}")
                return {"cids": top_cids}
                
            props = details_resp.json().get("PropertyTable", {}).get("Properties", [])
            
            # Construct a rich result
            result_context = {
                "source": "PubChem",
                "search_mz": mz,
                "compounds": props
            }
            return result_context

        except Exception as e:
            print(f"[PubChem] Exception during API call: {e}")
            return None
