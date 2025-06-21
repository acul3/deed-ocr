import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, List, Sequence, Optional
from datetime import datetime

import pandas as pd
import csv

# Import state abbreviation function from trs_validator
try:
    from deed_ocr.utils.trs_validator import TRSValidator
    # Create a standalone function that doesn't require class initialization
    def get_state_abbreviation_from_name(state_name: str) -> Optional[str]:
        """Get state abbreviation from state name using TRS validator logic."""
        if not state_name:
            return None
        # Default mapping for common states
        state_to_abbr = {
            "colorado": "CO",
            "wyoming": "WY",
            "nebraska": "NE", 
            "kansas": "KS",
            "new mexico": "NM",
            "utah": "UT",
            "oklahoma": "OK",
            "texas": "TX",
            "north dakota": "ND",
            "south dakota": "SD",
            "montana": "MT"
        }
        return state_to_abbr.get(state_name.lower())
except ImportError:
    # Fallback implementation if import fails
    def get_state_abbreviation_from_name(state_name: str) -> Optional[str]:
        """Fallback function to get state abbreviation from state name."""
        state_to_abbr = {
            "colorado": "CO",
            "wyoming": "WY", 
            "nebraska": "NE",
            "kansas": "KS",
            "new mexico": "NM",
            "utah": "UT",
            "oklahoma": "OK",
            "texas": "TX",
            "north dakota": "ND",
            "south dakota": "SD",
            "montana": "MT"
        }
        return state_to_abbr.get(state_name.lower()) if state_name else None

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def standardize_aliquot(aliquot_list):
    """
    Standardizes a list of aliquot strings by:
    - Replacing ½ and 1/2 and /2 with 2
    - Removing 1/4 and /4 suffixes
    - Removing spaces
    
    Args:
        aliquot_list: List of aliquot strings
    
    Returns:
        List of standardized aliquot strings
    
    Examples:
    ["W2NW1/4", "SE1/4NW1/4", "N2SW1/4", "SE1/4", "SW1/4NE1/4"] -> ["W2NW", "SENW", "N2SW", "SE", "SWNE"]
    ["S½"] -> ["S2"]
    ["E1/2NW1/4"] -> ["E2NW"]
    ["E/2NW/4"] -> ["E2NW"]
    ["E1/2 NW"] -> ["E2NW"]
    """
    standardized = []
    
    for aliquot_string in aliquot_list:
        # Replace fractions with 2
        aliquot = aliquot_string.replace('½', '2')
        aliquot = aliquot.replace('1/2', '2')
        aliquot = aliquot.replace('/2', '2')
        
        # Remove quarters (1/4 and /4)
        aliquot = aliquot.replace('1/4', '')
        aliquot = aliquot.replace('/4', '')
        
        standardized.append(aliquot)
    
    return standardized

def _extract_digits_and_orientation(value: str) -> tuple[str, str]:
    """Return (digits, orientation) from a Township / Range string.

    Examples:
    >>> _extract_digits_and_orientation('8N')
    ('8', 'N')
    >>> _extract_digits_and_orientation('56W')
    ('56', 'W')
    >>> _extract_digits_and_orientation('12')
    ('12', '')
    """
    if not value:
        return "", ""
    value = value.strip()
    digits_match = re.match(r"(\d+)", value)
    digits = digits_match.group(1) if digits_match else ""
    orientation = value[len(digits):] if digits else value  # remainder
    if orientation:
        orientation = orientation.strip().upper()
    return digits, orientation

def extract_first_number(text: str) -> str:
    """
    Extract the first number from a string and convert it to decimal.
    
    Args:
        text (str): Input string containing numbers
        
    Returns:
        str: Decimal representation of first number, or empty string if no number found
    """
    # Handle None or non-string inputs
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    
    # Pattern to match fractions (like 1/8, 12 1/2) or decimal numbers
    pattern = r'(\d+(?:\s+\d+\/\d+|\.\d+|\/\d+)|\d+)'
    
    match = re.search(pattern, text)
    
    if not match:
        return ""
    
    number_str = match.group(1).strip()
    
    try:
        # Handle mixed fractions (like "12 1/2")
        if ' ' in number_str and '/' in number_str:
            parts = number_str.split(' ')
            whole = int(parts[0])
            fraction_part = Fraction(parts[1])
            result = whole + fraction_part
        
        # Handle simple fractions (like "1/8")
        elif '/' in number_str:
            result = Fraction(number_str)
        
        # Handle decimal numbers
        else:
            result = float(number_str)
        
        return str(float(result))
    
    except (ValueError, ZeroDivisionError):
        return ""

def detect_oil(text: str) -> bool:
    """
    Detect 'oil' or 'oils' in text with various options.
    
    Args:
        text (str): Input string to check
        mode (str): Detection mode - 'oil', 'oils', 'any', or 'both'
        return_type (str): 'bool' for boolean result, 'type' for match type
        
    Returns:
        bool or str: Based on return_type parameter
    """
    # Handle None or non-string inputs
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    
    has_oil = bool(re.search(r'\boil\b', text, re.IGNORECASE))
    
    return has_oil

def _simplified_trs(township: str, rng: str, section: str) -> str:
    """Create simplified TRS string (e.g. T08N R56W S20).
    
    Format: T[township][orientation] R[range][orientation] S[section]
    Handle missing components gracefully.
    """
    parts = []
    
    # Handle Township
    if township:
        t_digits, t_ori = _extract_digits_and_orientation(township)
        if t_digits:
            # Pad single digits with leading zero
            if len(t_digits) == 1:
                t_digits = f"0{t_digits}"
            parts.append(f"T{t_digits}{t_ori}")
    
    # Handle Range
    if rng:
        r_digits, r_ori = _extract_digits_and_orientation(rng)
        if r_digits:
            # Pad single digits with leading zero
            if len(r_digits) == 1:
                r_digits = f"0{r_digits}"
            parts.append(f"R{r_digits}{r_ori}")
    
    # Handle Section
    if section:
        section_str = str(section).strip()
        if section_str:
            # Pad section to 2 digits
            section_digits = section_str.zfill(2)
            parts.append(f"S{section_digits}")
    
    return " ".join(parts)


def _format_date_mmdyyyy(date_str: str) -> str:
    """Format date string to MM/DD/YYYY format.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Formatted date string (MM/DD/YYYY) or empty string if invalid
    """
    if not date_str or not isinstance(date_str, str):
        return ""
    
    date_str = date_str.strip()
    if not date_str:
        return ""
    
    # Common date patterns to try
    date_patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # MM/DD/YYYY or MM-DD-YYYY
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})',  # MM/DD/YY or MM-DD-YY
        r'(\d{2})[/-](\d{2})[/-](\d{4})',      # DD/MM/YYYY or DD-MM-YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                groups = match.groups()
                
                # Try different interpretations based on pattern
                if len(groups[2]) == 4:  # Full year
                    if int(groups[0]) > 12:  # First group > 12, likely DD/MM/YYYY
                        day, month, year = groups
                    else:  # Assume MM/DD/YYYY
                        month, day, year = groups
                elif len(groups[0]) == 4:  # YYYY-MM-DD format
                    year, month, day = groups
                else:  # Two-digit year
                    month, day, year_short = groups
                    # Convert 2-digit year to 4-digit (assume 20XX for years < 50, 19XX for >= 50)
                    year_int = int(year_short)
                    year = str(2000 + year_int if year_int < 50 else 1900 + year_int)
                
                # Validate ranges
                month_int = int(month)
                day_int = int(day)
                year_int = int(year)
                
                if 1 <= month_int <= 12 and 1 <= day_int <= 31 and 1900 <= year_int <= 2100:
                    # Validate the date
                    datetime(year_int, month_int, day_int)
                    return f"{month_int:02d}/{day_int:02d}/{year_int}"
                    
            except (ValueError, TypeError):
                continue
    
    return ""  # Return empty string if date cannot be parsed


def _multiply_pad(num_str: str) -> str:
    """Multiply the numeric value by 10 and pad to 4 digits (used for TRS_Row_Code)."""
    if not num_str:
        return ""
    try:
        num_val = int(num_str) * 10
    except ValueError:
        return ""
    return str(num_val).zfill(4)


def _trs_row_code(details: Dict[str, Any], township: str, rng: str, section: str, county: str) -> str:
    """Create the TRS_Row_Code with state abbreviation.
    
    Format: [STATE_ABBR]_[county]:[township_code]-[range_code]-[section_code]
    """
    state = details.get("state", "")
    county = county
    
    # Get state abbreviation instead of full state name
    state_abbr = ""
    if state:
        state_abbr = get_state_abbreviation_from_name(state)
        if not state_abbr:
            state_abbr = state  # Fallback to original if abbreviation not found
    
    prefix = f"{state_abbr}_{county}:" if any([state_abbr, county]) else ""

    t_digits, t_ori = _extract_digits_and_orientation(township)
    r_digits, r_ori = _extract_digits_and_orientation(rng)

    t_code = f"{_multiply_pad(t_digits)}{t_ori}" if t_digits else ""
    r_code = f"{_multiply_pad(r_digits)}{r_ori}" if r_digits else ""
    s_code = str(section).zfill(3) if section else ""

    if not any([t_code, r_code, s_code]):
        return ""
    return f"{prefix}{t_code}-{r_code}-{s_code}"


def _generate_all_trs_row_codes(final_result: Dict[str, Any]) -> List[str]:
    """Generate all TRS_Row_Code values from the document's TRS_details.
    
    Args:
        final_result: The final result dictionary
        
    Returns:
        List of all TRS_Row_Code values in the document
    """
    trs_codes = []
    details = final_result.get("details", {}) if isinstance(final_result.get("details"), dict) else {}
    
    # Get all TRS_details entries
    trs_details_list = final_result.get("TRS_details", [])
    if not isinstance(trs_details_list, list):
        trs_details_list = [trs_details_list] if trs_details_list else []
    
    for trs_detail in trs_details_list:
        if isinstance(trs_detail, dict):
            township = trs_detail.get("Township", "")
            rng = trs_detail.get("Range", "")  
            section = trs_detail.get("Section", "")
            county = trs_detail.get("County", "")
            
            # Clean up township and range values
            if township:
                township = township.lower().replace("north", "N").replace("south", "S").replace(" ", "").upper()
            if rng:
                rng = rng.lower().replace("east", "E").replace("west", "W").replace(" ", "").upper()
            
            trs_code = _trs_row_code(details, township, rng, section, county)
            if trs_code and trs_code not in trs_codes:
                trs_codes.append(trs_code)
    
    return trs_codes


# ---------------------------------------------------------------------------
# Core conversion function
# ---------------------------------------------------------------------------

def convert_final_result_to_excel(
    final_result: Dict[str, Any],
    output_excel_path: Path,
    instruction_csv_path: Optional[Path] = None,
    column_order: Optional[Sequence[str]] = None,
) -> None:
    """Convert a *cleaned* final_result dictionary into an Excel file.

    Parameters
    ----------
    final_result: dict
        The contents of final_result.json already loaded in memory.
    output_excel_path: Path
        Destination path where the .xlsx file will be written.
    instruction_csv_path: Path | None
        Optional path to instruction.csv – used only to obtain column order.
    column_order: list[str] | None
        Explicit column order. Overrides *instruction_csv_path* if provided.
    """
    # Determine column order
    if column_order is not None:
        columns = list(column_order)
    elif instruction_csv_path and instruction_csv_path.exists():
        with instruction_csv_path.open(newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # skip header
            columns = [row[0] for row in reader if row]
    else:
        # Fallback to hard-coded order
        columns = [
            "Township",
            "Range",
            "Section",
            "Aliquot",
            "Simplified TRS",
            "TRS_Row_Code",
            "TRS_Doc_Code_List",
            "All_TRS_from_Document",
            "Document_Type",
            "Recording_Date",
            "Document_Date",
            "Effective_Date",
            "Granting_Language",
            "Lease_Royalty",
            "Lease_Royalty_Amended",
            "Lease_Primary_Term",
            "Book",
            "Page",
            "Ordered_Adjudged_Decreed",
            "Gross_Acreage",
            "Gross_Acreage_Check",
            "Lease Secondary_Term",
            "Reception Number",
            "Grantors_interest_mentioned",
            "Interest fraction",
            "Reservation",
            "Subject_to",
            "County",
            "State",
            "Grantor",
            "Grantee",
            "Grantor_address",
            "Grantee_address",
        ]

    # ------------------------------------------------------------------
    # Handle Grantor and Grantee arrays
    # ------------------------------------------------------------------
    
    details = final_result.get("details", {}) if isinstance(final_result.get("details"), dict) else {}
    
    # Extract grantor and grantee arrays
    grantor_list = details.get("grantor", [])
    grantee_list = details.get("grantee", [])
    
    # Ensure they are lists
    if not isinstance(grantor_list, list):
        grantor_list = [grantor_list] if grantor_list else []
    if not isinstance(grantee_list, list):
        grantee_list = [grantee_list] if grantee_list else []
    
    # Extract names and addresses from grantor/grantee objects
    grantor_names = []
    grantor_addresses = []
    grantee_names = []
    grantee_addresses = []
    
    for grantor in grantor_list:
        if isinstance(grantor, dict):
            name = grantor.get("name", "")
            address = grantor.get("address", "")
            grantor_names.append(str(name) if name else "")
            grantor_addresses.append(str(address) if address else "")
        elif grantor:
            grantor_names.append(str(grantor))
            grantor_addresses.append("")
    
    for grantee in grantee_list:
        if isinstance(grantee, dict):
            name = grantee.get("name", "")
            address = grantee.get("address", "")
            grantee_names.append(str(name) if name else "")
            grantee_addresses.append(str(address) if address else "")
        elif grantee:
            grantee_names.append(str(grantee))
            grantee_addresses.append("")

    # Prepare rows – one row per TRS_details entry (at least one row)
    trs_details_list = final_result.get("TRS_details", [])
    if not isinstance(trs_details_list, list):
        trs_details_list = [trs_details_list]
    if not trs_details_list:
        trs_details_list = [{}]

    rows: List[Dict[str, Any]] = []

    # Shared values across all rows
    details = final_result.get("details", {}) if isinstance(final_result.get("details"), dict) else {}
    lease_details = details.get("lease_details", {}) if isinstance(details.get("lease_details"), dict) else {}
    deed_details = details.get("deed_details", {}) if isinstance(details.get("deed_details"), dict) else {}
    gross_acreage = details.get("gross_acreage", "")

    # Keep these fields as lists instead of joining with semicolons
    legal_description_block_val = final_result.get("legal_description_block", [])
    if not isinstance(legal_description_block_val, list):
        legal_description_block_val = [str(legal_description_block_val)] if legal_description_block_val else []

    aliquot_val = final_result.get("aliquot", [])
    aliquot_val = standardize_aliquot(aliquot_val)
    if not isinstance(aliquot_val, list):
        if aliquot_val:
            aliquot = aliquot_val.replace('½', '2')
            aliquot = aliquot.replace('1/2', '2')
            aliquot = aliquot.replace('/2', '2')
            
            # Remove quarters (1/4 and /4)
            aliquot = aliquot.replace('1/4', '')
            aliquot = aliquot.replace('/4', '')
            aliquot_val = [str(aliquot_val)]  
        else: 
            aliquot_val = []

    interest_fraction_val = final_result.get("Interest_fraction", []) if not detect_oil(details.get("document_type", "")) else []
    if not isinstance(interest_fraction_val, list):
        interest_fraction_val = [str(interest_fraction_val)] if interest_fraction_val else []

    # Generate TRS_Doc_Code_List as array of all TRS_Row_Code values in the document
    trs_doc_code_list_val = _generate_all_trs_row_codes(final_result)

    reservation_val = final_result.get("reservation", [])
    if not isinstance(reservation_val, list):
        reservation_val = [str(reservation_val)] if reservation_val else []
    
    # Also handle Ordered_Adjudged_Decreed and Subject_to as lists for iteration
    ordered_adjudged_val = details.get("judgment_description", [])
    if not isinstance(ordered_adjudged_val, list):
        ordered_adjudged_val = [str(ordered_adjudged_val)] if ordered_adjudged_val else []
    
    subject_to_val = deed_details.get("subjecto", [])
    if not isinstance(subject_to_val, list):
        subject_to_val = [str(subject_to_val)] if subject_to_val else []

    lease_term = lease_details.get("lease_primary_term", lease_details.get("lease_term", ""))

    # Ensure lease_term is a string before regex processing
    if lease_term is None:
        lease_term = ""

    if not isinstance(lease_term, str):
        lease_term = str(lease_term)

    # Extract the first integer found in the lease term
    if lease_term:
        lease_term_match = re.search(r"\d+", lease_term)
        lease_term_number = lease_term_match.group(0) if lease_term_match else ""
    else:
        lease_term_number = ""
    
    gross_acreage_val = lease_details.get("gross_acreage", "") if lease_details.get("gross_acreage") else gross_acreage
    if gross_acreage_val != "":
        gross_acreage_val = re.search(r'\d+', str(gross_acreage_val))
        if gross_acreage_val:
            gross_acreage_val = gross_acreage_val.group(0)
        else:
            gross_acreage_val = ""
    else:
        gross_acreage_val = ""

    # ------------------------------ GRANTOR/GRANTEE ROWS ------------------------------
    # Calculate max rows needed for grantor/grantee arrays
    max_grantor_grantee_rows = max(len(grantor_names), len(grantor_addresses), len(grantee_names), len(grantee_addresses), 1)

    # ------------------------------------------------------------------
    # Compute total rows – consider TRS rows, party rows, and iterating field rows
    # ------------------------------------------------------------------
    num_trs_rows = len(trs_details_list)
    
    # Calculate max rows needed for iterating fields
    max_iterating_rows = max(
        len(aliquot_val),
        len(interest_fraction_val), 
        len(ordered_adjudged_val),
        len(reservation_val),
        len(subject_to_val),
        1  # At least 1 row
    )
    
    total_rows = max(num_trs_rows, max_grantor_grantee_rows, max_iterating_rows, 1)

    for row_index in range(total_rows):
        # ----------------------- TRS-specific fields -----------------------
        if row_index < num_trs_rows:
            trs = trs_details_list[row_index]
            township = trs.get("Township", "")
            rng = trs.get("Range", "")
            section = trs.get("Section", "")
            county = trs.get("County", "")
        else:
            township = rng = section = county = ""

        if township:
            township = township.lower().replace("north", "N").replace("south", "S").replace(" ", "").upper()
        if rng:
            rng = rng.lower().replace("east", "E").replace("west", "W").replace(" ", "").upper()

        # If no county in TRS details, use document-level county
        if not county and details:
            county = details.get("county", "")

        # Handle different field filling strategies
        
        row: Dict[str, Any] = {
            # TRS-specific (may be blank)
            "Township": township,
            "Range": rng,
            "Section": section,
            "Simplified TRS": _simplified_trs(township, rng, section),
            "TRS_Row_Code": _trs_row_code(details, township, rng, section, county),

            # ITERATING FIELDS - Fill each row with corresponding index value from list
            "Aliquot": aliquot_val[row_index] if row_index < len(aliquot_val) else "",
            "Granting_Language": interest_fraction_val[row_index] if row_index < len(interest_fraction_val) else "",
            "Interest fraction": interest_fraction_val[row_index] if row_index < len(interest_fraction_val) else "",
            "Ordered_Adjudged_Decreed": ordered_adjudged_val[row_index] if row_index < len(ordered_adjudged_val) else "",
            "Reservation": reservation_val[row_index] if row_index < len(reservation_val) else "",
            "Subject_to": subject_to_val[row_index] if row_index < len(subject_to_val) else "",
            
            # LIST FIELDS - Fill each row with the same complete list
            "TRS_Doc_Code_List": trs_doc_code_list_val,
            "All_TRS_from_Document": legal_description_block_val,
            
            # SHARED FIELDS - Fill all rows with the same value (for single values)
            "Document_Type": f'{details.get("document_type", "")}, {details.get("document_subtype", "")}',
            "Recording_Date": _format_date_mmdyyyy(details.get("recorded_date", "")),
            "Document_Date": _format_date_mmdyyyy(details.get("document_date", "")),
            "Effective_Date": _format_date_mmdyyyy(details.get("effective_date", "")),
            "Lease_Royalty": extract_first_number(lease_details.get("lease_royalty", "none")),
            "Lease_Royalty_Amended": lease_details.get("lease_royalty", ""),
            "Lease_Primary_Term": lease_term_number,
            "Book": details.get("book_county", ""),
            "Page": details.get("page_county", ""),
            "Gross_Acreage": lease_details.get("gross_acreage", "") if lease_details.get("gross_acreage") else gross_acreage,
            "Gross_Acreage_Check": True if lease_details.get("gross_acreage") else "None",
            "Lease Secondary_Term": lease_details.get("lease_secondary_lease_term", ""),
            "Reception Number": details.get("reception_number", ""),
            "Grantors_interest_mentioned": deed_details.get("grantors_interest", ""),
            "County": details.get("county", ""),
            "State": details.get("state", ""),
        }

        # ----------------------- Grantor/Grantee columns (index-wise) -----------------------
        row["Grantor"] = grantor_names[row_index] if row_index < len(grantor_names) else ""
        row["Grantee"] = grantee_names[row_index] if row_index < len(grantee_names) else ""
        row["Grantor_address"] = grantor_addresses[row_index] if row_index < len(grantor_addresses) else ""
        row["Grantee_address"] = grantee_addresses[row_index] if row_index < len(grantee_addresses) else ""

        # Ensure order & include only requested columns
        ordered_row = {col: row.get(col, "") for col in columns}
        rows.append(ordered_row)

    # Create DataFrame & write
    df = pd.DataFrame(rows, columns=columns)
    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_excel_path, index=False, engine="openpyxl")


def combine_excel_files_to_index(
    results_dir: Path,
    output_index_path: Path,
    instruction_csv_path: Optional[Path] = None
) -> None:
    """
    Combine all final_result.xlsx files from processed PDFs into a single Index_Output.xlsx.
    
    Args:
        results_dir: Directory containing PDF result folders
        output_index_path: Path for the combined Index_Output.xlsx
        instruction_csv_path: Optional path to instruction.csv for column order
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Find all final_result.xlsx files
    excel_files = list(results_dir.rglob("final_result.xlsx"))
    
    if not excel_files:
        logger.warning(f"No final_result.xlsx files found in {results_dir}")
        return
    
    logger.info(f"Found {len(excel_files)} Excel files to combine")
    
    # Read all Excel files and combine them
    all_dataframes = []
    
    for excel_file in excel_files:
        try:
            df = pd.read_excel(excel_file, engine="openpyxl")
            if not df.empty:
                # Add document name column to identify source
                document_name = excel_file.parent.name  # Get folder name (PDF name)
                df.insert(0, 'Document_Name', document_name)
                all_dataframes.append(df)
                logger.info(f"Added {len(df)} rows from {document_name}")
        except Exception as e:
            logger.error(f"Error reading {excel_file}: {str(e)}")
            continue
    
    if not all_dataframes:
        logger.error("No valid Excel files could be read")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    # Ensure output directory exists
    output_index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save combined Excel file
    combined_df.to_excel(output_index_path, index=False, engine="openpyxl")
    
    logger.info(f"Combined {len(all_dataframes)} Excel files into {output_index_path}")
    logger.info(f"Total rows in combined file: {len(combined_df)}")


def collect_and_rename_text_files(
    results_dir: Path,
    doc_texts_dir: Path
) -> None:
    """
    Collect all full_text.txt files and rename them to {pdf_name}_{reception_number}.txt
    in the doc_texts subfolder.
    
    Args:
        results_dir: Directory containing PDF result folders
        doc_texts_dir: Directory to save renamed text files
    """
    import logging
    import json
    logger = logging.getLogger(__name__)
    
    # Create doc_texts directory
    doc_texts_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF result folders
    pdf_folders = [d for d in results_dir.iterdir() if d.is_dir()]
    
    collected_count = 0
    
    for pdf_folder in pdf_folders:
        full_text_file = pdf_folder / "full_text.txt"
        final_result_file = pdf_folder / "final_result.json"
        
        if not full_text_file.exists():
            logger.warning(f"No full_text.txt found in {pdf_folder.name}")
            continue
            
        # Get PDF name from folder name
        pdf_name = pdf_folder.name
        
        # Try to get reception number from final_result.json
        reception_number = None
        if final_result_file.exists():
            try:
                with open(final_result_file, 'r', encoding='utf-8') as f:
                    final_result = json.load(f)
                
                # Extract reception number from details
                details = final_result.get("details", {})
                if isinstance(details, dict):
                    reception_number = details.get("reception_number")
                    
            except Exception as e:
                logger.warning(f"Could not read final_result.json from {pdf_folder.name}: {str(e)}")
        
        # Create new filename
        if reception_number:
            new_filename = f"{pdf_name}_{reception_number}.txt"
        else:
            new_filename = f"{pdf_name}_no_reception.txt"
            logger.warning(f"No reception number found for {pdf_name}, using {new_filename}")
        
        # Copy and rename the file
        try:
            new_file_path = doc_texts_dir / new_filename
            
            # Read content and write to new location
            with open(full_text_file, 'r', encoding='utf-8') as source:
                content = source.read()
            
            with open(new_file_path, 'w', encoding='utf-8') as dest:
                dest.write(content)
            
            collected_count += 1
            logger.info(f"Copied {pdf_folder.name}/full_text.txt -> {new_filename}")
            
        except Exception as e:
            logger.error(f"Error copying text file from {pdf_folder.name}: {str(e)}")
    
    logger.info(f"Collected {collected_count} text files in {doc_texts_dir}") 