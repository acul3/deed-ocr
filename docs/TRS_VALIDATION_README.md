# TRS Validation

This document explains the TRS (Township, Range, Section) validation functionality in the deed-ocr system.

## Overview

The TRS validation feature validates extracted Township, Range, and Section details against the official PLSS (Public Land Survey System) geodatabase to ensure accuracy and detect potential OCR errors.

## Features

- ✅ Validates Township and Range combinations against PLSS geodatabase
- ✅ Validates Section numbers within valid Township/Range combinations  
- ✅ Supports county-to-state mapping for proper validation
- ✅ Generates Excel reports with validation results
- ✅ Integrated into the main workflow
- ✅ Handles missing data gracefully

## Requirements

### Required Dependencies

```bash
pip install geopandas>=0.14.0
```

### Required Data Files

1. **PLSS Geodatabase (.gdb)**: Official geodatabase containing PLSS data
   - Should contain `PLSSTownship` and `PLSSFirstDivision` layers
   - Example path: `/path/to/ilmocplss.gdb`

2. **Counties JSON (Optional)**: County to state mapping file
   - Format: `[{"County": "Adams County", "Abbreviation": "CO"}, ...]`
   - If not provided, uses built-in mapping for common Colorado counties

## Usage

### Environment Variables (Recommended)

Set up TRS validation paths in your `.env` file for automatic use:

```bash
# Add to .env file
GEODATABASE_PATH=/path/to/plss.gdb
COUNTIES_JSON_PATH=/path/to/counties.json
```

Then use without specifying paths:
```bash
# TRS validation automatically enabled if GEODATABASE_PATH is set
deed-ocr -i input.pdf -o ./results

# Process multiple files with automatic TRS validation
deed-ocr -i ./pdf_folder -o ./results --recursive
```

### Command Line Interface

You can also provide paths directly via command line (overrides environment variables):

```bash
# Basic usage with TRS validation
deed-ocr -i input.pdf -o ./results --geodatabase-path /path/to/plss.gdb

# With counties mapping file
deed-ocr -i input.pdf -o ./results \
  --geodatabase-path /path/to/plss.gdb \
  --counties-json-path /path/to/counties.json

# Disable TRS validation (even if geodatabase provided)
deed-ocr -i input.pdf -o ./results --disable-trs-validation
```

### Python API

```python
from pathlib import Path
from deed_ocr.workflow import process_deed_pdf_simple

# Process with TRS validation
result = process_deed_pdf_simple(
    pdf_path=Path("deed.pdf"),
    output_dir=Path("./results"),
    geodatabase_path="/path/to/plss.gdb",
    counties_json_path="/path/to/counties.json",  # Optional
    enable_trs_validation=True
)
```

### Direct TRS Validation

```python
from deed_ocr.utils.trs_validator import TRSValidator, validate_trs_from_final_result

# Validate from existing final result
validation_results = validate_trs_from_final_result(
    Path("./results/document/final_result.json"),
    geodatabase_path="/path/to/plss.gdb",
    output_excel_path=Path("./trs_validation.xlsx")
)

# Manual validation
validator = TRSValidator("/path/to/plss.gdb")
trs_data = {
    "Township": "5 North",
    "Range": "66 West", 
    "Section": "29",
    "County": "Weld"
}
result = validator.validate_trs(trs_data)
print(f"Valid: {result.is_valid}")
```

## Output Files

When TRS validation is enabled, the following additional file is created:

```
results/
└── document_name/
    ├── final_result.json
    ├── final_result.xlsx
    ├── trs_validation.xlsx  ← TRS validation results
    └── ...
```

### TRS Validation Excel Format

The `trs_validation.xlsx` file contains the following columns:

| Column | Description |
|--------|-------------|
| Document | Document name |
| TRS_Index | Index of TRS entry within document |
| Township | Township value (e.g., "5 North") |
| Range | Range value (e.g., "66 West") |
| Section | Section number (e.g., "29") |
| County | County name(s) (e.g., "Weld" or "Weld, Morgan") |
| County_Count | Number of counties to validate |
| TRS | Combined TRS string (e.g., "T5N R66W S29") |
| is_valid_or_invalid | "Valid", "Invalid", "Valid (for N counties)", etc. |
| log | Detailed validation log messages |
| error_message | Error details if validation failed |

## Validation Logic

The validation process follows these steps:

1. **Extract TRS Details**: From `TRS_details` in final result
2. **Multiple County Handling**: Split comma-separated counties (e.g., "Weld, Morgan")
3. **County to State Mapping**: Convert each county name to state abbreviation
4. **Format Processing**: Clean and standardize Township/Range strings
5. **Multi-County Validation**: Try validation against each county's state
6. **Township/Range Validation**: Query geodatabase for valid combination
7. **Section Validation**: If section provided, validate against PLSS divisions
8. **Result Generation**: Create validation result with status and logs

### Multiple Counties Support

The validator handles TRS entries that span multiple counties:

- **Input**: `"County": "Weld, Morgan"` or `"County": "Weld, Morgan County"`
- **Processing**: Validates against both Weld County (CO) and Morgan County (CO)
- **Result**: Valid if **any** county validates successfully
- **Logging**: Shows which counties passed/failed validation

**Examples:**
```json
{
  "County": "Weld, Morgan",           // ✅ Validates both counties
  "County": "Weld, Morgan County",    // ✅ Handles "County" suffix  
  "County": "Weld, InvalidCounty"     // ✅ Valid if Weld validates (partial success)
}
```

### Validation Results

- **Valid**: Township, Range, and Section (if provided) exist in PLSS data for **at least one** county
- **Valid (for N counties)**: Multiple counties provided, successful validation for N counties
- **Invalid**: TRS components don't exist in PLSS data for **any** of the provided counties
- **Invalid (checked N counties)**: Multiple counties provided, all failed validation
- **No TRS**: No TRS details found in document

**Multi-County Logic**: A TRS entry is considered **valid** if it validates successfully against any of the provided counties. This handles cases where land parcels span county boundaries.

## Error Handling

The system gracefully handles various error conditions:

- **Missing GeoPandas**: Validation disabled with warning
- **Missing Geodatabase**: Validation disabled with error message
- **Invalid TRS Data**: Marked as invalid with descriptive error
- **Database Query Errors**: Logged with technical details

## Testing

Use the provided test script to verify TRS validation:

```bash
cd scripts
python test_trs_validation.py
```

The test script will:
- Test validation against existing final result files
- Test manual validation with sample data
- Generate validation Excel reports
- Display validation results and summaries

## Configuration

### Default County Mapping

If no counties JSON file is provided, the system includes built-in mapping for common Colorado counties:

```python
county_to_state = {
    "adams": "CO",
    "weld": "CO", 
    "larimer": "CO",
    "boulder": "CO",
    # ... etc
}
```

### Geodatabase Requirements

The PLSS geodatabase must contain:

- **PLSSTownship** layer with fields:
  - `STATEABBR`: State abbreviation
  - `TWNSHPNO`: Township number (3 digits, zero-padded)
  - `TWNSHPDIR`: Township direction (N/S)
  - `RANGENO`: Range number (3 digits, zero-padded) 
  - `RANGEDIR`: Range direction (E/W)
  - `PLSSID`: Unique identifier

- **PLSSFirstDivision** layer with fields:
  - `PLSSID`: Township identifier
  - `FRSTDIVNO`: Section number (2 digits, zero-padded)

## Troubleshooting

### Common Issues

1. **GeoPandas Import Error**
   ```
   Solution: pip install geopandas
   ```

2. **Geodatabase Not Found**
   ```
   Solution: Verify the geodatabase path exists and is accessible
   ```

3. **No Matching Township**
   ```
   Possible causes:
   - OCR error in Township/Range extraction
   - County not mapped to correct state
   - Township/Range combination doesn't exist in PLSS data
   ```

4. **Invalid Section**
   ```
   Possible causes:
   - OCR error in section number
   - Section doesn't exist in the specified township
   ```

### Debug Logging

Enable verbose logging to see detailed validation steps:

```bash
deed-ocr -i input.pdf -o ./results --geodatabase-path /path/to/plss.gdb --verbose
```

## Performance Notes

- **Geodatabase is loaded once at startup** for optimal performance
- TRS validation adds ~1-2 seconds per document
- No performance penalty for reprocessing multiple documents in batch
- Validation is only performed if TRS details are found
- Failed validations don't stop document processing
- Memory usage is optimized with single geodatabase instance 