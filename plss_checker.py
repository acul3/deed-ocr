import geopandas as gpd

file_path = "/Users/samsulrahmadani/Downloads/geojson/ilmocplss.gdb"  # Replace with your file path (e.g., GeoPackage, FileGDB)

layers = gpd.list_layers(file_path)
#open_layer = "PLSSTownship"  # Replace with the layer you want to open
gdf_township = gpd.read_file(file_path, layer="PLSSTownship")
gdf_first_division = gpd.read_file(file_path, layer="PLSSFirstDivision")

#load json county /Users/samsulrahmadani/Documents/deed-ocr/counties_list.json
import json
with open("/Users/samsulrahmadani/Documents/deed-ocr/counties_list.json", "r") as f:
    counties = json.load(f)
    
# create function to get state from county name
def get_state_from_county(county_name):
    #remove county name suffix
    county_name = county_name.replace(' County', '')
    for g in counties:
        if g['County'].replace(' County', '').lower() == county_name.lower():
            return g['Abbreviation']
    return None

import re
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
        orientation = orientation.upper()  # Ensure orientation is uppercase
    return digits, orientation

# #crete this function township = gdf_township.gdf_township.query(
#     "STATEABBR=='CO' & "
#     "TWNSHPNO=='010' & TWNSHPDIR=='N' & "
#     "RANGENO=='058' & RANGEDIR=='W'"
# )

def get_township(gdf_township,gdf_first_division, trs):
    county = trs.get('County', None)
    range_ = trs.get('Range', None)
    if county is None:
        print("County not provided in TRS details.")
        return False
    elif range_ is None:
        print("Range not provided in TRS details.")
        return False
    state_abbr = get_state_from_county(trs['County'])
    #get township number and direction using regex
    # Example: "8 North" -> "008 N"
    # Example: "57 West" -> "057 W"
    township = trs['Township'].lower().replace("north", "N").replace("south", "S").replace(" ", "")
    range_ = trs['Range'].lower().replace("west", "W").replace("east", "E").replace(" ", "")
    township_number, township_direction = _extract_digits_and_orientation(township)
    range_number, range_direction = _extract_digits_and_orientation(range_)
    section = trs.get('Section', None)
    print(f"Extracted Township: {township_number} {township_direction}, Range: {range_number} {range_direction}")
    township = gdf_township.query(
        "STATEABBR==@state_abbr & "
        "TWNSHPNO==@township_number.zfill(3) & "
        "TWNSHPDIR==@township_direction & "
        "RANGENO==@range_number.zfill(3) & "
        "RANGEDIR==@range_direction.strip()"
    )
    plssid = township.get('PLSSID').values
    # If PLSSID is empty, return None
    if plssid.size == 0:
        print("No matching township found.")
        return False
    else:
        print(f"Found matching township with PLSSID: {plssid[0]}")
        plssid_number = plssid[0].strip()
        if section:
            # If section is provided, filter by section
            section_value = gdf_first_division.query(f"PLSSID == '{plssid_number}' and FRSTDIVNO=='{section.zfill(2)}'").copy()
            if section_value.empty:
                print("Township and Range Valid But Section Invalid")
                return False
            else:
                print(f"Township , Range and Section Valid")
                return True
        else:
            # If no section is provided, return the township
            print("PTownship and Range Valid BUT no Section provided.")
            return True
        
        
        
# #Input sample:
# TRS_Detail = [{
#     "County": "Adams County",
#     "Township": "8 North",
#     "Range": "57 West",
#     "Section": "12"
# }]

#INPUT json: the final_result from the json file
# import json
# with open("/Users/samsulrahmadani/Documents/deed-ocr/results_oil_deed/1170220-1/final_result.json", "r") as f:
#     final_result = json.load(f)
#     TRS_Detail = final_result['TRS_details']

#extract trs from all json
import glob
page_json = glob.glob("/Users/samsulrahmadani/Documents/deed-ocr/results_stage_2/*/pages/*.json")
import json
TRS_Detail = []
for g in page_json:
    with open(g, "r") as f:
        final_result = json.load(f)
        if final_result.get("TRS_details",None):
            TRS_Detail.extend(final_result['TRS_details'])

for dic in TRS_Detail:
    trs = dic
    print("=============")
    print(f"Processing TRS: {trs}")
    result = get_township(gdf_township, gdf_first_division, trs)
    print("++++++++++++++++")