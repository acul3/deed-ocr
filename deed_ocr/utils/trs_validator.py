"""TRS Validation utility for validating Township, Range, Section details against PLSS geodatabase."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import StringIO
import sys

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("GeoPandas not available. TRS validation will be disabled.")


class TRSValidationResult:
    """Result of TRS validation."""
    
    def __init__(self, is_valid: bool, log_messages: List[str], error_message: str = "", plssid: str = ""):
        self.is_valid = is_valid
        self.log_messages = log_messages
        self.error_message = error_message
        self.plssid = plssid
    
    def __str__(self):
        return f"Valid: {self.is_valid}, Messages: {'; '.join(self.log_messages)}, PLSSID: {self.plssid}"


class TRSValidator:
    """Validates TRS (Township, Range, Section) details against PLSS geodatabase."""
    
    def __init__(self, geodatabase_path: Optional[str] = None, counties_json_path: Optional[str] = None):
        """
        Initialize TRS validator.
        
        Args:
            geodatabase_path: Path to PLSS geodatabase file (.gdb)
            counties_json_path: Path to counties list JSON file for state mapping
        """
        self.geodatabase_path = geodatabase_path
        self.counties_json_path = counties_json_path
        self.gdf_township = None
        self.gdf_first_division = None
        self.counties_data = None
        self.is_initialized = False
        
        if not GEOPANDAS_AVAILABLE:
            logger.error("GeoPandas is not available. TRS validation cannot be performed.")
            return
            
        if geodatabase_path:
            self._load_geodatabase()
        if counties_json_path:
            self._load_counties_data()
    
    def _load_geodatabase(self):
        """Load PLSS geodatabase files."""
        try:
            if not Path(self.geodatabase_path).exists():
                logger.error(f"Geodatabase file not found: {self.geodatabase_path}")
                return
                
            logger.info(f"Loading PLSS geodatabase from: {self.geodatabase_path}")
            self.gdf_township = gpd.read_file(self.geodatabase_path, layer="PLSSTownship")
            self.gdf_first_division = gpd.read_file(self.geodatabase_path, layer="PLSSFirstDivision")
            logger.info("Successfully loaded PLSS geodatabase")
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Error loading geodatabase: {str(e)}")
            self.is_initialized = False
    
    def _load_counties_data(self):
        """Load counties data for state mapping."""
        try:
            if not Path(self.counties_json_path).exists():
                logger.warning(f"Counties JSON file not found: {self.counties_json_path}")
                return
                
            with open(self.counties_json_path, 'r') as f:
                self.counties_data = json.load(f)
            logger.info(f"Loaded {len(self.counties_data)} counties from {self.counties_json_path}")
            
        except Exception as e:
            logger.error(f"Error loading counties data: {str(e)}")
            self.counties_data = None
    
    def get_state_from_county(self, county_name: str) -> Optional[str]:
        """
        Get state abbreviation from county name.
        
        Args:
            county_name: County name (e.g., "Adams County")
            
        Returns:
            State abbreviation (e.g., "CO") or None if not found
        """
        if not self.counties_data:
            # Default mapping for common counties if counties data not available
            county_to_state = {
                "adams": "CO",
                "weld": "CO",
                "larimer": "CO",
                "boulder": "CO",
                "jefferson": "CO",
                "arapahoe": "CO",
                "douglas": "CO",
                "el paso": "CO",
                "pueblo": "CO",
                "mesa": "CO"
            }
            clean_county = county_name.replace(' County', '').lower()
            return county_to_state.get(clean_county)
        
        # Remove county suffix for comparison
        clean_county_name = county_name.replace(' County', '')
        
        for county_data in self.counties_data:
            if county_data.get('County', '').replace(' County', '').lower() == clean_county_name.lower():
                return county_data.get('Abbreviation')
        
        return None
    
    def get_state_abbreviation_from_name(self, state_name: str) -> Optional[str]:
        """
        Get state abbreviation from state name.
        
        Args:
            state_name: State name (e.g., "Colorado", "Wyoming")
            
        Returns:
            State abbreviation (e.g., "CO", "WY") or None if not found
        """
        if not self.counties_data:
            # Default mapping for common states if counties data not available
            state_to_abbr = {
                "colorado": "CO",
                "wyoming": "WY",
                "nebraska": "NE", 
                "kansas": "KS",
                "new mexico": "NM",
                "utah": "UT",
                "oklahoma": "OK"
            }
            return state_to_abbr.get(state_name.lower())
        
        # Find state abbreviation from counties data
        state_name_lower = state_name.lower()
        for county_data in self.counties_data:
            # Check if this county's state matches
            state_field = county_data.get('State', '').lower()
            if state_field == state_name_lower:
                return county_data.get('Abbreviation')
        
        return None
    
    def _extract_digits_and_orientation(self, value: str) -> Tuple[str, str]:
        """
        Extract digits and orientation from a Township/Range string.
        
        Args:
            value: Township or Range string (e.g., "8N", "56W", "12")
            
        Returns:
            Tuple of (digits, orientation)
        """
        if not value:
            return "", ""
            
        value = value.strip()
        digits_match = re.match(r"(\d+)", value)
        digits = digits_match.group(1) if digits_match else ""
        orientation = value[len(digits):] if digits else value
        if orientation:
            orientation = orientation.upper()
        
        return digits, orientation
    
    def validate_trs(self, trs_details: Dict[str, Any], document_state: Optional[str] = None) -> TRSValidationResult:
        """
        Validate a single TRS details entry.
        
        Args:
            trs_details: Dictionary containing Township, Range, Section, County
            document_state: Optional state name from document details (e.g., "Colorado")
            
        Returns:
            TRSValidationResult with validation status and log messages
        """
        # Capture print statements for logging
        old_stdout = sys.stdout
        sys.stdout = string_buffer = StringIO()
        
        log_messages = []
        
        try:
            if not GEOPANDAS_AVAILABLE:
                return TRSValidationResult(
                    False, 
                    ["GeoPandas not available - cannot perform validation"],
                    "GeoPandas not available"
                )
            
            if not self.is_initialized:
                return TRSValidationResult(
                    False, 
                    ["TRS Validator not initialized - geodatabase not loaded"],
                    "Validator not initialized"
                )
            
            # Validate required fields
            county = trs_details.get('County', None)
            range_ = trs_details.get('Range', None)
            township = trs_details.get('Township', None)
            section = trs_details.get('Section', None)
            
            if county is None:
                log_messages.append("County not provided in TRS details.")
                return TRSValidationResult(False, log_messages, "Missing county")
            
            if range_ is None:
                log_messages.append("Range not provided in TRS details.")
                return TRSValidationResult(False, log_messages, "Missing range")
            
            if township is None:
                log_messages.append("Township not provided in TRS details.")
                return TRSValidationResult(False, log_messages, "Missing township")
            
            # Handle multiple counties separated by commas
            counties = [c.strip() for c in county.split(',')]
            log_messages.append(f"Found {len(counties)} county/counties: {', '.join(counties)}")
            
            # Check if document state is provided and get its abbreviation
            document_state_abbr = None
            if document_state:
                document_state_abbr = self.get_state_abbreviation_from_name(document_state)
                if document_state_abbr:
                    log_messages.append(f"Using document state: {document_state} -> {document_state_abbr}")
                else:
                    log_messages.append(f"Could not get abbreviation for document state: {document_state}")
            
            # Try validation against each county
            validation_results = []
            successful_counties = []
            found_plssid = ""
            
            for county_name in counties:
                # Clean county name (remove "County" suffix if present)
                clean_county_name = county_name.replace(' County', '').strip()
                
                # Determine state abbreviation - prioritize document state
                state_abbr = None
                if document_state_abbr:
                    # Use document state if available
                    state_abbr = document_state_abbr
                    log_messages.append(f"Using document state ({document_state_abbr}) for county: {clean_county_name}")
                else:
                    # Fall back to county-based state lookup
                    state_abbr = self.get_state_from_county(clean_county_name)
                    if state_abbr:
                        log_messages.append(f"Determined state from county: {clean_county_name} -> {state_abbr}")
                    else:
                        log_messages.append(f"Could not determine state for county: {clean_county_name}")
                        continue
                
                log_messages.append(f"Trying validation with county: {clean_county_name} (state: {state_abbr})")
                
                # Process township and range strings
                township_clean = township.lower().replace("north", "N").replace("south", "S").replace(" ", "")
                range_clean = range_.lower().replace("west", "W").replace("east", "E").replace(" ", "")
                
                township_number, township_direction = self._extract_digits_and_orientation(township_clean)
                range_number, range_direction = self._extract_digits_and_orientation(range_clean)
                
                # Query the geodatabase for this state
                try:
                    query_result = self.gdf_township.query(
                        "STATEABBR==@state_abbr & "
                        "TWNSHPNO==@township_number.zfill(3) & "
                        "TWNSHPDIR==@township_direction & "
                        "RANGENO==@range_number.zfill(3) & "
                        "RANGEDIR==@range_direction.strip()"
                    )
                    
                    plssid = query_result.get('PLSSID').values if not query_result.empty else []
                    
                    # Check if township/range combination exists for this county/state
                    if len(plssid) == 0:
                        log_messages.append(f"No matching township found for {clean_county_name} County ({state_abbr})")
                        continue
                    
                    log_messages.append(f"Found matching township for {clean_county_name} County with PLSSID: {plssid[0]}")
                    plssid_number = plssid[0].strip()
                    if not found_plssid:  # Store the first PLSSID found
                        found_plssid = plssid_number
                    
                    # If section is provided, validate it too
                    if section:
                        section_query = self.gdf_first_division.query(
                            f"PLSSID == '{plssid_number}' and FRSTDIVNO=='{str(section).zfill(2)}'"
                        )
                        
                        if section_query.empty:
                            log_messages.append(f"Township and Range valid for {clean_county_name} County but Section {section} invalid")
                            continue
                        else:
                            log_messages.append(f"Township, Range and Section valid for {clean_county_name} County")
                            successful_counties.append(clean_county_name)
                    else:
                        log_messages.append(f"Township and Range valid for {clean_county_name} County (no section provided)")
                        successful_counties.append(clean_county_name)
                        
                except Exception as e:
                    log_messages.append(f"Error querying geodatabase for {clean_county_name} County: {str(e)}")
                    continue
            
            # Summary of validation results
            if not successful_counties:
                log_messages.append(f"TRS validation failed for all counties: {', '.join(counties)}")
                return TRSValidationResult(False, log_messages, "No valid counties found")
            else:
                log_messages.append(f"TRS validation successful for: {', '.join(successful_counties)}")
                if len(successful_counties) < len(counties):
                    failed_counties = [c for c in counties if c.replace(' County', '').strip() not in successful_counties]
                    log_messages.append(f"TRS validation failed for: {', '.join(failed_counties)}")
                return TRSValidationResult(True, log_messages, plssid=found_plssid)
                
        except Exception as e:
            log_messages.append(f"Validation error: {str(e)}")
            return TRSValidationResult(False, log_messages, f"Validation error: {str(e)}")
        
        finally:
            # Restore stdout and capture any print statements
            sys.stdout = old_stdout
            printed_output = string_buffer.getvalue()
            if printed_output.strip():
                log_messages.extend(printed_output.strip().split('\n'))
    
    def validate_trs_list(self, trs_details_list: List[Dict[str, Any]], document_state: Optional[str] = None) -> List[TRSValidationResult]:
        """
        Validate a list of TRS details.
        
        Args:
            trs_details_list: List of TRS details dictionaries
            document_state: Optional state name from document details (e.g., "Colorado")
            
        Returns:
            List of TRSValidationResult objects
        """
        results = []
        for i, trs_details in enumerate(trs_details_list):
            logger.info(f"Validating TRS {i+1}/{len(trs_details_list)}: {trs_details}")
            result = self.validate_trs(trs_details, document_state)
            results.append(result)
            
        return results


def create_trs_validation_excel(
    trs_details_list: List[Dict[str, Any]], 
    validation_results: List[TRSValidationResult],
    output_excel_path: Path,
    document_name: str = ""
) -> None:
    """
    Create Excel file with TRS validation results.
    
    Args:
        trs_details_list: List of TRS details dictionaries
        validation_results: List of corresponding validation results
        output_excel_path: Path to save Excel file
        document_name: Name of the document being validated
    """
    rows = []
    
    for i, (trs_details, validation_result) in enumerate(zip(trs_details_list, validation_results)):
        # Handle multiple counties in the display
        county_str = trs_details.get("County", "")
        
        # Determine validation status
        status = "Valid" if validation_result.is_valid else "Invalid"
        
        row = {
            "Document": document_name,
            "TRS_Index": i + 1,
            "Township": trs_details.get("Township", ""),
            "Range": trs_details.get("Range", ""),
            "Section": trs_details.get("Section", ""),
            "County": county_str,
            "TRS": trs_details.get("TRS", ""),
            "is_valid_or_invalid": status,
            "log": "; ".join(validation_result.log_messages),
            "error_message": validation_result.error_message,
            "PLSSID": validation_result.plssid
        }
        rows.append(row)
    
    # If no TRS details, create a row indicating no TRS found
    if not rows:
        rows.append({
            "Document": document_name,
            "TRS_Index": 0,
            "Township": "",
            "Range": "",
            "Section": "",
            "County": "",
            "TRS": "",
            "is_valid_or_invalid": "No TRS",
            "log": "No TRS details found in document",
            "error_message": "No TRS details extracted",
            "PLSSID": ""
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(rows)
    
    # Ensure output directory exists
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    df.to_excel(output_excel_path, index=False, engine="openpyxl")
    logger.info(f"TRS validation results saved to: {output_excel_path}")


def validate_trs_from_final_result(
    final_result_path: Path,
    geodatabase_path: Optional[str] = None,
    counties_json_path: Optional[str] = None,
    output_excel_path: Optional[Path] = None
) -> List[TRSValidationResult]:
    """
    Validate TRS from a final_result.json file.
    
    Args:
        final_result_path: Path to final_result.json file
        geodatabase_path: Path to PLSS geodatabase
        counties_json_path: Path to counties JSON file
        output_excel_path: Optional path to save Excel validation results
        
    Returns:
        List of TRSValidationResult objects
    """
    # Load final result
    with open(final_result_path, 'r', encoding='utf-8') as f:
        final_result = json.load(f)
    
    # Extract TRS details
    trs_details_list = final_result.get("TRS_details", [])
    if not isinstance(trs_details_list, list):
        trs_details_list = [trs_details_list] if trs_details_list else []
    
    # Extract document state from details
    document_state = None
    if "details" in final_result and isinstance(final_result["details"], dict):
        document_state = final_result["details"].get("state")
    
    # Initialize validator
    validator = TRSValidator(geodatabase_path, counties_json_path)
    
    # Validate TRS with document state
    validation_results = validator.validate_trs_list(trs_details_list, document_state)
    
    # Create Excel if path provided
    if output_excel_path:
        document_name = final_result_path.parent.name
        create_trs_validation_excel(
            trs_details_list, 
            validation_results, 
            output_excel_path, 
            document_name
        )
    
    return validation_results 