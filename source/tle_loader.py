import json
import requests
import re


def save_tle(progress_callback=None):

    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get('content-length', 0))
        ESTIMATED_TOTAL = 10 * 1024 * 1024
        downloaded = 0
        chunks = []

        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                if progress_callback:
                    effective_total = total if total > 0 else ESTIMATED_TOTAL
                    progress_callback(downloaded, effective_total)

        raw = b"".join(chunks).decode("utf-8")
        # strip() removes \r
        data = [line.strip() for line in raw.splitlines() if line.strip()]
        with open("data.json", "w", encoding="utf-8") as file:
            json.dump(data, file)
        return len(data) // 3
    except Exception as e:
        raise RuntimeError(str(e))


def dataset_sat(number_sat):

    with open("data.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    tle_text = []
    tle_num = 0
    DO_NOT_OPEN = data
    line_1 = line_2 = ""
    line_0 = ""

    for i in range(number_sat, number_sat + 3):
        if tle_num == 0:
            line_0 = DO_NOT_OPEN[i]
        if tle_num == 1:
            line_1 = DO_NOT_OPEN[i]
        if tle_num == 2:
            line_2 = DO_NOT_OPEN[i]
        tle_text.append(DO_NOT_OPEN[i])
        tle_num = tle_num + 1

    return (line_0, line_1, line_2)


def find_satellite_by_norad_id(norad_id):

    try:
        with open("data.json", "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading data.json: {e}")
        return None

    norad_str = str(norad_id).strip()

    # iterate through data in groups of 3 lines
    for i in range(0, len(data) - 2, 3):
        name_line = data[i].strip()
        line1 = data[i + 1].strip()
        line2 = data[i + 2].strip()

        if not line1.startswith('1 ') or not line2.startswith('2 '):
            continue

        # NORAD ID is in line1 positions 2-7
        try:
            tle_norad = line1[2:7].strip()
            if tle_norad == norad_str:
                return (name_line, line1, line2)
        except (IndexError, ValueError):
            continue

    return None


def parse_tle_line(line: str):

    tokens = []
    numbers_list = []
    pattern = re.compile(r'\s+|[+-]?\d+(?:\.\d+)?')
    number_index = 0

    for m in pattern.finditer(line):
        text = m.group()

        if text.isspace():
            tokens.append(text)
        else:
            value = float(text) if "." in text else int(text)
            token = [value, text, number_index]
            tokens.append(token)
            numbers_list.append(token)
            number_index += 1

    return tokens, numbers_list


def update_number(token, new_value):
    old_text = token[1]

    if "." in old_text:
        decimals = len(old_text.split(".")[1])
        token[1] = f"{new_value:.{decimals}f}"
        token[0] = float(new_value)
    else:
        width = len(old_text)
        token[1] = f"{int(new_value):0{width}d}"
        token[0] = int(new_value)


def build_tle_line(tokens):
    return "".join(
        t if isinstance(t, str) else t[1]
        for t in tokens
    )


def parse_tle_fields(line1, line2):
    fields = {}

    # From line 1
    fields['satellite_number'] = line1[2:7].strip()
    fields['classification'] = line1[7].strip()
    fields['international_designator'] = line1[9:17].strip()
    fields['epoch_year'] = line1[18:20].strip()
    fields['epoch_day'] = line1[20:32].strip()
    fields['first_derivative'] = line1[33:43].strip()
    fields['second_derivative'] = line1[44:52].strip()
    fields['bstar'] = line1[53:61].strip()
    fields['ephemeris_type'] = line1[62].strip()
    fields['element_number'] = line1[64:68].strip()

    # From line 2
    fields['inclination'] = line2[8:16].strip()
    fields['raan'] = line2[17:25].strip()
    fields['eccentricity'] = line2[26:33].strip()
    fields['argument_perigee'] = line2[34:42].strip()
    fields['mean_anomaly'] = line2[43:51].strip()
    fields['mean_motion'] = line2[52:63].strip()
    fields['revolution_number'] = line2[63:68].strip()

    return fields


def tle_checksum(line: str) -> int:
    """compute TLE line checksum"""
    total = 0
    for c in line[:68]:  # first 68 characters, excluding the last
        if c.isdigit():
            total += int(c)
        elif c == '-':
            total += 1
    return total % 10


def fix_checksum(line: str) -> str:
    """return TLE line with corrected checksum"""
    return line[:68] + str(tle_checksum(line))