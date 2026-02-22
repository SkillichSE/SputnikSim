# parse_tle.py
"""TLE parser. Extracts orbital parameters and classifies orbit type"""

import numpy as np
from datetime import datetime, timedelta

EARTH_RADIUS = 6378.137    # km
MU_EARTH     = 398600.4418 # km^3/s^2


class TLEParseError(Exception):
    pass


def parse_tle(name, line1, line2):
    """Parse a TLE triplet. Returns vector, tle_struct"""
    try:
        if len(line1) < 69 or len(line2) < 69:
            raise TLEParseError("TLE lines are too short (minimum 69 characters each)")
        if line1[0] != '1' or line2[0] != '2':
            raise TLEParseError("Invalid TLE line identifiers (expected '1' and '2')")

        # Line 1
        catalog_number  = line1[2:7].strip()
        classification  = line1[7]
        intl_designator = line1[9:17].strip()

        year_str = line1[18:20].strip()
        if not year_str.isdigit():
            raise TLEParseError("Invalid epoch year in TLE line 1")
        year = int(year_str)
        year = 2000 + year if year < 57 else 1900 + year  # 00-56 → 2000s, 57-99 → 1900s

        day_of_year    = float(line1[20:32].strip())
        epoch_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        mean_motion_dot  = float(line1[33:43].strip())                    # rev/day^2
        mean_motion_ddot = parse_scientific_notation(line1[44:52].strip()) # rev/day^3
        bstar            = parse_scientific_notation(line1[53:61].strip()) # 1/earth_radii

        ephemeris_type = line1[62].strip()
        element_number = int(line1[64:68].strip())

        # Line 2
        inclination      = float(line2[8:16].strip())          # deg
        raan             = float(line2[17:25].strip())         # deg
        eccentricity     = float("0." + line2[26:33].strip())  # implied decimal point
        argument_perigee = float(line2[34:42].strip())         # deg
        mean_anomaly     = float(line2[43:51].strip())         # deg
        mean_motion      = float(line2[52:63].strip())         # rev/day
        revolution_number = int(line2[63:68].strip())

        vector = [inclination, raan, eccentricity, argument_perigee, mean_anomaly, mean_motion]

        tle_struct = {
            "name":             name.strip(),
            "catalog_number":   catalog_number,
            "classification":   classification,
            "intl_designator":  intl_designator,
            "epoch": {
                "year":        year,
                "day_of_year": day_of_year,
                "datetime":    epoch_datetime,
            },
            "mean_motion_dot":   mean_motion_dot,
            "mean_motion_ddot":  mean_motion_ddot,
            "bstar":             bstar,
            "ephemeris_type":    ephemeris_type,
            "element_number":    element_number,
            "inclination":       inclination,
            "raan":              raan,
            "eccentricity":      eccentricity,
            "argument_perigee":  argument_perigee,
            "mean_anomaly":      mean_anomaly,
            "mean_motion":       mean_motion,
            "revolution_number": revolution_number,
            "line1":             line1.strip(),
            "line2":             line2.strip(),
        }

        return vector, tle_struct

    except (ValueError, IndexError) as e:
        raise TLEParseError(f"TLE parse error: {e}")


def parse_scientific_notation(s):
    """Parse TLE packed scientific notation 12345-3 → 0.00012345,  -12345-3 → -0.00012345"""
    if not s or s.isspace():
        return 0.0
    s = s.strip()
    if not s or s == '0':
        return 0.0

    import re
    m = re.fullmatch(r'([+-]?)(\d+)([+-]\d)', s)
    if m:
        sign_str = m.group(1)
        digits   = m.group(2)
        exp_str  = m.group(3)
        sign     = -1.0 if sign_str == '-' else 1.0
        mantissa = sign * float('0.' + digits)
        return mantissa * (10 ** int(exp_str))

    try:
        return float(s)
    except ValueError:
        return 0.0


def calculate_orbital_period(mean_motion):
    """Return orbital period in minutes"""
    if mean_motion == 0:
        return 0.0
    return 1440.0 / mean_motion


def calculate_semi_major_axis(mean_motion):
    """Return semi-major axis in km via Kepler's third law"""
    if mean_motion == 0:
        return 0.0
    n = mean_motion * 2 * np.pi / 86400.0  # rev/day → rad/s
    return (MU_EARTH / (n ** 2)) ** (1.0 / 3.0)


def calculate_apogee_perigee(semi_major_axis, eccentricity):
    """Return altitudes above surface in km"""
    apogee  = semi_major_axis * (1 + eccentricity) - EARTH_RADIUS
    perigee = semi_major_axis * (1 - eccentricity) - EARTH_RADIUS
    return apogee, perigee


def get_orbit_type(mean_motion, inclination, eccentricity):
    """Classify orbit"""
    sma      = calculate_semi_major_axis(mean_motion)
    altitude = sma - EARTH_RADIUS

    if eccentricity >= 0.25:
        return "HEO"

    if abs(mean_motion - 1.0) <= 0.15:
        if inclination < 5.0 and eccentricity < 0.01:
            return "GEO"
        return "GSO"

    if altitude < 2000:
        if 97.5 <= inclination <= 99.5:
            return "SSO"
        if inclination > 80:
            return "LEO_POLAR"
        return "LEO"

    if 2000 <= altitude <= 35786:
        return "MEO"

    return "OTHER"


def classify_orbit(inclination, mean_motion, eccentricity):
    """Return human readable label"""
    label = get_orbit_type(mean_motion, inclination, eccentricity)
    labels = {
        "GEO":       "GEO (Geostationary)",
        "GSO":       "GSO (Geosynchronous Inclined)",
        "HEO":       "HEO (Highly Elliptical)",
        "MEO":       "MEO (Medium Earth Orbit)",
        "LEO":       "LEO (Low Earth Orbit)",
        "LEO_POLAR": "LEO Polar",
        "SSO":       "SSO (Sun-Synchronous)",
        "OTHER":     "Other / Deep Space",
    }
    return labels.get(label, "Unknown")


def is_geo_orbit(mean_motion, tolerance=0.15):
    """Return True if mean_motion is in the GEO/GSO band"""
    return abs(mean_motion - 1.0) <= tolerance


if __name__ == "__main__":
    tests = [
        ("ISS (ZARYA)",
         "1 25544U 98067A   24001.50000000  .00012345  00000-0  12345-3 0  9992",
         "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391111111"),
    ]

    for tname, l1, l2 in tests:
        try:
            vector, tle_data = parse_tle(tname, l1, l2)
            print(f"Name:    {tle_data['name']}")
            print(f"Vector:  {vector}")
            print(f"Orbit:   {classify_orbit(tle_data['inclination'], tle_data['mean_motion'], tle_data['eccentricity'])}")
            print(f"Period:  {calculate_orbital_period(tle_data['mean_motion']):.2f} min")
            sma = calculate_semi_major_axis(tle_data['mean_motion'])
            apo, per = calculate_apogee_perigee(sma, tle_data['eccentricity'])
            print(f"Apogee:  {apo:.1f} km")
            print(f"Perigee: {per:.1f} km")
        except TLEParseError as e:
            print(f"Error: {e}")