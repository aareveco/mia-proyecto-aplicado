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
        Search for compounds by converting input m/z to potential neutral masses (Adducts).
        Considers: Neutral (M), Protonated [M+H]+, Deprotonated [M-H]-
        """
        # Common Adducts (Mass diff from Neutral M)
        # [M+H]+ : M = mz - 1.007276
        # [M-H]- : M = mz + 1.007276
        # Neutral: M = mz
        
        proton_mass = 1.007276
        
        potential_masses = [
            {"label": "Neutral (M)", "mass": mz},
            {"label": "[M+H]+", "mass": mz - proton_mass},
            {"label": "[M-H]-", "mass": mz + proton_mass}
        ]

        all_results = []
        
        for p in potential_masses:
            target_mass = p["mass"]
            if target_mass <= 0:
                continue
                
            min_mz = target_mass - tolerance
            max_mz = target_mass + tolerance
            
            # URL: .../compound/monoisotopic_mass/range/{min}/{max}/cids/JSON
            search_url = f"{self.BASE_URL}/compound/monoisotopic_mass/range/{min_mz}/{max_mz}/cids/JSON"
            
            try:
                response = requests.get(search_url, timeout=20)
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get("IdentifierList", {}).get("CID", [])
                    if cids:
                        # Take top 2 per adduct to keep it diverse but concise
                        for cid in cids[:2]:
                            all_results.append({"cid": cid, "adduct": p["label"]})
            except Exception as e:
                print(f"[PubChem] Error searching mass {target_mass} ({p['label']}): {e}")

        if not all_results:
            return None
            
        # Deduplicate by CID
        unique_cids = {}
        for item in all_results:
            if item["cid"] not in unique_cids:
                unique_cids[item["cid"]] = item["adduct"]
        
        # Limit total results
        final_cids = list(unique_cids.keys())[:5]
        
        if not final_cids:
            return None

        # 2. Fetch Details
        cids_str = ",".join(map(str, final_cids))
        details_url = f"{self.BASE_URL}/compound/cid/{cids_str}/property/Title,MolecularFormula/JSON"
        
        try:
            details_resp = requests.get(details_url, timeout=20)
            if details_resp.status_code != 200:
                return {"cids": final_cids}
                
            props = details_resp.json().get("PropertyTable", {}).get("Properties", [])
            
            # Add Adduct info and Bioassays
            # Only fetch bioassays for the very first match to save time
            bio_fetched = False
            for p in props:
                cid = p.get("CID")
                # Add Adduct Label
                p["Adduct"] = unique_cids.get(cid, "Unknown")
                
                if not bio_fetched:
                    bio_info = self._fetch_bioassays(cid)
                    p["Bioactivity"] = bio_info
                    bio_fetched = True
                else:
                    p["Bioactivity"] = []

            result_context = {
                "source": "PubChem",
                "search_mz": mz,
                "compounds": props
            }
            return result_context

        except Exception as e:
            print(f"[PubChem] Exception fetching details: {e}")
            return None

    def _fetch_bioassays(self, cid: int, limit: int = 5) -> List[str]:
        """
        Fetch active bioassays for a given CID.
        URL: .../compound/cid/{cid}/assaysummary/JSON
        """
        url = f"{self.BASE_URL}/compound/cid/{cid}/assaysummary/JSON"
        try:
            # Short timeout, optional feature
            resp = requests.get(url, timeout=5)
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
