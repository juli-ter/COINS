from pathlib import Path
import ast
import csv
import shutil

RAW_ROOT = Path(r"C:\Users\labri\Uni\Thesis\COINS_Claude\analysis\python_port\rawData")

TARGET_COLUMNS = [
    "blockID", "currentFrame", "laserRotation", "shieldRotation",
    "shieldDegrees", "currentHit", "totalReward", "sendTrigger",
    "triggerValue", "trueMean", "trueVariance", "volatility", "eyePosition"
]


def is_already_normal_csv(csv_path: Path) -> bool:
    """
    Normal file like sub-01 starts with:
    blockID,currentFrame,...
    Bad file starts with:
    ['blockID', ...] or [blockID, ...]
    """
    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline().strip()

    return first_line.startswith("blockID,currentFrame")


def parse_value(value: str):
    value = value.strip()

    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("None", "nan", "NaN", ""):
        return ""

    # remove quotes around strings
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        return value[1:-1]

    # try int
    try:
        if "." not in value:
            return int(value)
    except ValueError:
        pass

    # try float
    try:
        return float(value)
    except ValueError:
        return value


def parse_weird_line(line: str):
    """
    Handles both:
    ['blockID', 'currentFrame', ...]
    [blockID, currentFrame, ...]
    [2, 0, 64.0, 360, 20, False, ...]
    """
    line = line.strip()

    # first try normal Python list parsing
    try:
        parsed = ast.literal_eval(line)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # fallback for malformed list without quotes in header
    if line.startswith("[") and line.endswith("]"):
        line = line[1:-1]

    reader = csv.reader([line], skipinitialspace=True)
    parts = next(reader)

    return [parse_value(part) for part in parts]


def convert_file(csv_path: Path):
    if is_already_normal_csv(csv_path):
        print(f"SKIP already OK: {csv_path}")
        return

    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")

    if not backup_path.exists():
        shutil.copy2(csv_path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    with csv_path.open("r", encoding="utf-8-sig", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"EMPTY FILE, skip: {csv_path}")
        return

    rows = []

    # We do not trust the old header. We force the correct header like in sub-01.
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            row = parse_weird_line(line)
        except Exception as e:
            print(f"ERROR in file {csv_path}, line {line_number}")
            print(f"Line content: {line[:300]}")
            raise e

        # Bad files often have only 12 columns, missing eyePosition.
        # Add empty eyePosition to match sub-01 format.
        if len(row) < len(TARGET_COLUMNS):
            row = row + [""] * (len(TARGET_COLUMNS) - len(row))

        # If somehow there are too many columns, cut extra columns.
        row = row[:len(TARGET_COLUMNS)]

        rows.append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TARGET_COLUMNS)
        writer.writerows(rows)

    print(f"CONVERTED: {csv_path}")


def main():
    for sub_num in range(10, 23):
        sub_dir = RAW_ROOT / f"sub-{sub_num:02d}"
        beh_dir = sub_dir / "ses-2-meg" / "beh"

        if not beh_dir.exists():
            print(f"Missing beh folder: {beh_dir}")
            continue

        csv_files = sorted(beh_dir.glob("*.csv"))

        # Do not process backups
        csv_files = [p for p in csv_files if not p.name.endswith(".bak")]

        if len(csv_files) != 4:
            print(f"WARNING: {beh_dir} has {len(csv_files)} csv files, expected 4")

        for csv_path in csv_files:
            convert_file(csv_path)


if __name__ == "__main__":
    main()