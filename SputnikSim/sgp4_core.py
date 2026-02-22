import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from math import sin, cos, atan2, sqrt, degrees, radians

EARTH_RADIUS = 6378.137
MU_EARTH = 398600.4418


# Earth rotation

def gmst_from_jd(jd):
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = (
            67310.54841
            + (876600 * 3600 + 8640184.812866) * T
            + 0.093104 * T ** 2
            - 6.2e-6 * T ** 3
    )
    gmst_rad = (gmst_sec % 86400.0) * (2 * np.pi / 86400.0)
    return gmst_rad


# TEME to ECEF conversion

def teme_to_ecef(r_teme, jd):
    theta = gmst_from_jd(jd)

    R = np.array([
        [cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])

    return R @ r_teme


# ECEF to geographic coordinates

def ecef_to_latlon(r_ecef):
    x, y, z = r_ecef

    lon = atan2(y, x)
    lat = atan2(z, sqrt(x ** 2 + y ** 2))

    return degrees(lat), degrees(lon)


def calculate_orbital_parameters(r_teme, v_teme):

    r = np.array(r_teme)
    v = np.array(v_teme)

    # vector magnitudes
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)

    # angular momentum vector
    h = np.cross(r, v)
    h_mag = np.linalg.norm(h)

    # semimajor axis
    energy = (v_mag ** 2) / 2 - MU_EARTH / r_mag
    a = -MU_EARTH / (2 * energy)

    # eccentricity vector
    e_vec = ((v_mag ** 2 - MU_EARTH / r_mag) * r - np.dot(r, v) * v) / MU_EARTH
    e = np.linalg.norm(e_vec)

    # inclination
    i = degrees(np.arccos(h[2] / h_mag))

    # apogee and perigee
    apogee = a * (1 + e) - EARTH_RADIUS
    perigee = a * (1 - e) - EARTH_RADIUS

    # orbital period
    period = 2 * np.pi * sqrt(a ** 3 / MU_EARTH) / 60  # in minutes

    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': i,
        'apogee_km': apogee,
        'perigee_km': perigee,
        'period_min': period,
        'altitude_km': r_mag - EARTH_RADIUS
    }


def time_now():
    import time
    from datetime import datetime
    date = time.ctime(time.time())
    date = list(map(str, date.split()))
    now = datetime.now()
    today = str(now.date())
    hours, minutes, seconds = [int(x) for x in date[3].split(":")]
    year, month, day = [int(x) for x in today.split("-")]
    return (year, month, day, hours, minutes, seconds)


def simulate(tle_line1, tle_line2, start_time, duration_hours=24, step_minutes=1.0):

    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    results = []
    current_time = start_time

    # statistics
    min_altitude = float('inf')
    max_altitude = 0
    total_distance = 0
    prev_r_ecef = None
    error_count = 0

    while current_time <= start_time + timedelta(hours=duration_hours):
        jd, fr = jday(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second
        )

        error, r_teme, v_teme = sat.sgp4(jd, fr)

        if error == 0:
            r_teme = np.array(r_teme)
            v_teme = np.array(v_teme)

            # ECEF coordinates
            r_ecef = teme_to_ecef(r_teme, jd + fr)
            lat, lon = ecef_to_latlon(r_ecef)

            # orbital parameters
            orbital_params = calculate_orbital_parameters(r_teme, v_teme)

            # compute distance traveled
            if prev_r_ecef is not None:
                distance = np.linalg.norm(r_ecef - prev_r_ecef)
                total_distance += distance
            prev_r_ecef = r_ecef

            # update statistics
            altitude = orbital_params['altitude_km']
            min_altitude = min(min_altitude, altitude)
            max_altitude = max(max_altitude, altitude)

            results.append({
                'time': current_time,
                'timestamp': current_time.isoformat(),

                # TEME coordinates
                'x_teme': r_teme[0],
                'y_teme': r_teme[1],
                'z_teme': r_teme[2],

                # ECEF coordinates
                'x_ecef': r_ecef[0],
                'y_ecef': r_ecef[1],
                'z_ecef': r_ecef[2],

                # geographic coordinates
                'lat': lat,
                'lon': lon,

                # velocity
                'vx': v_teme[0],
                'vy': v_teme[1],
                'vz': v_teme[2],
                'velocity_kmps': np.linalg.norm(v_teme),

                # orbital parameters
                'altitude_km': altitude,
                'apogee_km': orbital_params['apogee_km'],
                'perigee_km': orbital_params['perigee_km'],
                'semi_major_axis': orbital_params['semi_major_axis'],
                'eccentricity': orbital_params['eccentricity'],
                'inclination': orbital_params['inclination'],
                'period_min': orbital_params['period_min'],

                'error': error
            })
        else:
            error_count += 1
            results.append({
                'time': current_time,
                'timestamp': current_time.isoformat(),
                'error': error
            })

        current_time += timedelta(minutes=step_minutes)

    # summary statistics
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': (start_time + timedelta(hours=duration_hours)).isoformat(),
        'duration_hours': duration_hours,
        'step_minutes': step_minutes,
        'total_points': len(results),
        'successful_points': len(results) - error_count,
        'error_points': error_count,
        'min_altitude_km': min_altitude if min_altitude != float('inf') else 0,
        'max_altitude_km': max_altitude,
        'total_distance_km': total_distance,
        'avg_altitude_km': (min_altitude + max_altitude) / 2 if min_altitude != float('inf') else 0,
    }

    # add start end position info
    if len(results) > 0 and 'lat' in results[0]:
        summary['start_position'] = {
            'lat': results[0]['lat'],
            'lon': results[0]['lon'],
            'altitude_km': results[0]['altitude_km']
        }

        if 'lat' in results[-1]:
            summary['end_position'] = {
                'lat': results[-1]['lat'],
                'lon': results[-1]['lon'],
                'altitude_km': results[-1]['altitude_km']
            }

            # compute position drift
            lat_drift = results[-1]['lat'] - results[0]['lat']
            lon_drift = results[-1]['lon'] - results[0]['lon']
            summary['position_drift'] = {
                'lat_deg': lat_drift,
                'lon_deg': lon_drift
            }

    return results, summary


def print_simulation_summary(summary, console_output=True):

    output = []
    output.append("=" * 60)
    output.append("SIMULATION SUMMARY")
    output.append("=" * 60)
    output.append(f"Duration:        {summary['duration_hours']:.1f} hours")
    output.append(f"Step:            {summary['step_minutes']:.1f} minutes")
    output.append(f"Total points:    {summary['total_points']}")
    output.append(f"Successful:      {summary['successful_points']}")

    if summary['error_points'] > 0:
        output.append(f"Errors:          {summary['error_points']}")

    output.append("")
    output.append("ORBITAL PARAMETERS:")
    output.append(f"Min altitude:    {summary['min_altitude_km']:.2f} km")
    output.append(f"Max altitude:    {summary['max_altitude_km']:.2f} km")
    output.append(f"Avg altitude:    {summary['avg_altitude_km']:.2f} km")
    output.append(f"Total distance:  {summary['total_distance_km']:.2f} km")

    if 'start_position' in summary:
        output.append("")
        output.append("POSITION:")
        sp = summary['start_position']
        output.append(f"Start:  LAT {sp['lat']:7.3f}°  LON {sp['lon']:8.3f}°  ALT {sp['altitude_km']:.2f} km")

        if 'end_position' in summary:
            ep = summary['end_position']
            output.append(f"End:    LAT {ep['lat']:7.3f}°  LON {ep['lon']:8.3f}°  ALT {ep['altitude_km']:.2f} km")

            if 'position_drift' in summary:
                drift = summary['position_drift']
                output.append(f"Drift:  LAT {drift['lat_deg']:+7.3f}°  LON {drift['lon_deg']:+8.3f}°")

    output.append("=" * 60)

    result = "\n".join(output)

    if console_output:
        print(result)

    return result