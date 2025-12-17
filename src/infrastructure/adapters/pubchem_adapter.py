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
            
            # --- NEW: Fetch BioAssays for the top 1 compound to add bioactivity context ---
            # We only do it for the first one to save time/bandwidth in this demo
            if top_cids:
                main_cid = top_cids[0]
                bio_info = self._fetch_bioassays(main_cid)
                # Attach to the first property object if it matches CID (it should)
                for p in props:
                    if p.get("CID") == main_cid:
                        p["Bioactivity"] = bio_info
                        break

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

    def _fetch_bioassays(self, cid: int, limit: int = 5) -> List[str]:
        """
        Fetch active bioassays for a given CID.
        URL: .../compound/cid/{cid}/assaysummary/JSON
        """
        url = f"{self.BASE_URL}/compound/cid/{cid}/assaysummary/JSON"
        try:
            # Short timeout, optional feature
            resp = requests.get(url, timeout=3)
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            # Debug Print
            # print(f"[DEBUG] PubChem Assay Data Keys: {data.keys()}")
            
            table = data.get("Table", {}).get("Row", [])
            
            activities = []
            for row in table:
                cell = row.get("Cell", [])
                
                # Cell is a list of strings: [AID, ..., Activity, ..., AssayName, ...]
                # From debug: Index 4 is Activity, Index 9 is Name (usually)
                # But to be safe vs schema changes, we can look for "Active" and then find the longest string or the one at index 9.
                
                # Ensure cell has enough items
                if len(cell) < 10:
                    continue
                    
                activity_outcome = cell[4] if len(cell) > 4 else ""
                assay_name = cell[9] if len(cell) > 9 else ""
                
                if activity_outcome == "Active" and assay_name:
                    activities.append(assay_name)
            
            # Deduplicate and limit
            unique_acts = list(set(activities))
            return unique_acts[:limit]
            
        except Exception as e:
            # print(f"[PubChem] Error fetching bioassays for {cid}: {e}")
            return []
