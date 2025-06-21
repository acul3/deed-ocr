import json
from pathlib import Path


def main():
    """Merge state abbreviations into counties list.

    1. Read state_abbre.json which contains objects with `name` and `abbreviation`.
    2. Read counties_list.json which contains objects with `County` and `State` keys.
    3. For every entry in counties list, add an `Abbreviation` field derived from matching `State`.
    4. Ensure that each state from state_abbre.json appears at least once in the counties list; if a state is absent, append a stub record with an empty county string.
    5. Write the merged result back to counties_list.json (overwriting) **and** create a timestamped backup of the original file alongside it (``counties_list.backup.json``).
    """

    root = Path(__file__).resolve().parent.parent  # repository root

    state_abbre_path = root / "state_abbre.json"
    counties_list_path = root / "counties_list.json"

    # Backup original counties file
    backup_path = counties_list_path.with_suffix(".backup.json")
    if not backup_path.exists():
        backup_path.write_bytes(counties_list_path.read_bytes())
        print(f"Backup of counties list written to {backup_path.relative_to(root)}")

    # Load both JSON files
    with state_abbre_path.open("r", encoding="utf-8") as f:
        states = json.load(f)

    with counties_list_path.open("r", encoding="utf-8") as f:
        counties = json.load(f)

    # Build mapping of state name -> abbreviation
    name_to_abbr = {state_obj["name"]: state_obj["abbreviation"] for state_obj in states}

    # Track which states are present in counties list
    present_states = set()

    # Augment existing counties entries
    for entry in counties:
        state_name = entry["State"]
        present_states.add(state_name)
        # Add or overwrite abbreviation
        entry["Abbreviation"] = name_to_abbr.get(state_name, "")

    # Append stub entries for states not yet represented
    missing_states = [s for s in name_to_abbr.keys() if s not in present_states]
    if missing_states:
        for state_name in missing_states:
            counties.append({
                "County": "",
                "State": state_name,
                "Abbreviation": name_to_abbr[state_name]
            })
        print(f"Added {len(missing_states)} stub record(s) for states missing in counties list.")

    # Write back updated counties list (pretty-printed for readability)
    with counties_list_path.open("w", encoding="utf-8") as f:
        json.dump(counties, f, indent=2, ensure_ascii=False)
    print(f"Updated counties list written to {counties_list_path.relative_to(root)} (total rows: {len(counties)})")


if __name__ == "__main__":
    main() 