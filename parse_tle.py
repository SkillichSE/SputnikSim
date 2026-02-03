"""
Парсер TLE (Two-Line Element) данных с семантическим анализом
Извлекает орбитальные параметры и определяет тип объекта (спутник/мусор)
"""
import numpy as np
from datetime import datetime, timedelta
import re
from enum import Enum


class ObjectType(Enum):
    """Типы космических объектов"""
    SATELLITE = "satellite"  # Активный/неактивный спутник
    DEBRIS = "debris"  # Космический мусор
    ROCKET_BODY = "rocket_body"  # Ступень ракеты
    FRAGMENT = "fragment"  # Осколок
    UNKNOWN = "unknown"  # Неопределенный


class OperationalStatus(Enum):
    """Эксплуатационный статус"""
    ACTIVE = "active"  # Активный
    INACTIVE = "inactive"  # Неактивный
    DEAD = "dead"  # Потерян
    DECAYED = "decayed"  # Сгорел в атмосфере
    FRAGMENTED = "fragmented"  # Фрагментирован
    UNKNOWN = "unknown"  # Неизвестен


class TLEParseError(Exception):
    """Исключение для ошибок парсинга TLE"""
    pass


def classify_object_type(name):
    """
    Классифицирует объект по названию

    Args:
        name: название объекта из TLE

    Returns:
        ObjectType: тип объекта
    """
    name_upper = name.upper()

    # Маркеры мусора
    debris_markers = ['DEB', 'DEBRIS']
    rocket_markers = ['R/B', 'ROCKET BODY', 'RB', 'BOOSTER']
    fragment_markers = ['FRAG', 'FRAGMENT', 'FRAG.']

    # Проверка маркеров
    for marker in debris_markers:
        if marker in name_upper:
            return ObjectType.DEBRIS

    for marker in rocket_markers:
        if marker in name_upper:
            return ObjectType.ROCKET_BODY

    for marker in fragment_markers:
        if marker in name_upper:
            return ObjectType.FRAGMENT

    # Если нет маркеров мусора - считаем спутником
    return ObjectType.SATELLITE


def check_operational_status_by_name(name):
    """
    Проверяет статус по ключевым словам в названии

    Args:
        name: название объекта

    Returns:
        OperationalStatus: предполагаемый статус
    """
    name_upper = name.upper()

    # Маркеры неактивных объектов
    if any(marker in name_upper for marker in ['DEAD', 'DEFUNCT', 'RETIRED']):
        return OperationalStatus.DEAD

    if any(marker in name_upper for marker in ['DECAYED', 'REENTERED']):
        return OperationalStatus.DECAYED

    if any(marker in name_upper for marker in ['FRAG', 'EXPLOSION', 'COLLISION']):
        return OperationalStatus.FRAGMENTED

    # По умолчанию - неизвестен
    return OperationalStatus.UNKNOWN


def parse_tle(name, line1, line2):
    """
    Парсит TLE данные и возвращает числовой вектор и структурированные данные

    Args:
        name: Название объекта (строка 0)
        line1: Первая строка TLE
        line2: Вторая строка TLE

    Returns:
        tuple: (vector, tle_struct)
            - vector: список из 6 орбитальных параметров
            - tle_struct: словарь со всеми извлеченными данными
    """
    try:
        # Проверка базовой структуры
        if len(line1) < 69 or len(line2) < 69:
            raise TLEParseError("TLE строки слишком короткие")

        if line1[0] != '1' or line2[0] != '2':
            raise TLEParseError("Неверный формат TLE строк")

        # ========== Парсинг Line 1 ==========

        # Номер каталога (NORAD ID)
        catalog_number = line1[2:7].strip()

        # Классификация
        classification = line1[7]

        # Международное обозначение (COSPAR ID)
        intl_designator = line1[9:17].strip()

        # Epoch (время элементов)
        year_str = line1[18:20].strip()
        if not year_str.isdigit():
            raise TLEParseError("Некорректный год в TLE")

        year = int(year_str)
        year = 2000 + year if year < 57 else 1900 + year

        day_of_year_str = line1[20:32].strip()
        day_of_year = float(day_of_year_str)

        # Конвертация в datetime
        epoch_datetime = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

        # Первая производная среднего движения
        mean_motion_dot = float(line1[33:43].strip())

        # Вторая производная среднего движения
        mean_motion_ddot_str = line1[44:52].strip()
        mean_motion_ddot = parse_scientific_notation(mean_motion_ddot_str)

        # BSTAR drag term
        bstar_str = line1[53:61].strip()
        bstar = parse_scientific_notation(bstar_str)

        # Номер элемента
        element_number = int(line1[64:68].strip())

        # ========== Парсинг Line 2 ==========

        # Наклонение (inclination) в градусах
        inclination = float(line2[8:16].strip())

        # Прямое восхождение восходящего узла (RAAN) в градусах
        raan = float(line2[17:25].strip())

        # Эксцентриситет (без начального нуля)
        eccentricity_str = line2[26:33].strip()
        eccentricity = float("0." + eccentricity_str)

        # Аргумент перигея в градусах
        argument_perigee = float(line2[34:42].strip())

        # Средняя аномалия в градусах
        mean_anomaly = float(line2[43:51].strip())

        # Среднее движение (обороты/день)
        mean_motion = float(line2[52:63].strip())

        # Номер оборота на эпохе
        revolution_number = int(line2[63:68].strip())

        # ========== Классификация объекта ==========

        object_type = classify_object_type(name)
        status_by_name = check_operational_status_by_name(name)

        # ========== Формирование выходных данных ==========

        # Вектор для модели (6 основных орбитальных параметров)
        vector = [
            inclination,
            raan,
            eccentricity,
            argument_perigee,
            mean_anomaly,
            mean_motion
        ]

        # Полная структура данных
        tle_struct = {
            "name": name.strip(),
            "catalog_number": catalog_number,
            "classification": classification,
            "intl_designator": intl_designator,
            "epoch": {
                "year": year,
                "day_of_year": day_of_year,
                "datetime": epoch_datetime
            },
            "mean_motion_dot": mean_motion_dot,
            "mean_motion_ddot": mean_motion_ddot,
            "bstar": bstar,
            "element_number": element_number,
            "inclination": inclination,
            "raan": raan,
            "eccentricity": eccentricity,
            "argument_perigee": argument_perigee,
            "mean_anomaly": mean_anomaly,
            "mean_motion": mean_motion,
            "revolution_number": revolution_number,
            "line1": line1.strip(),
            "line2": line2.strip(),
            # Новые поля
            "object_type": object_type,
            "status_by_name": status_by_name
        }

        return vector, tle_struct

    except (ValueError, IndexError) as e:
        raise TLEParseError(f"Ошибка парсинга TLE: {str(e)}")


def parse_scientific_notation(s):
    """Парсит TLE-нотацию вида 12345-3 -> 0.00012345"""
    s = s.strip()
    if not s or s.isspace():
        return 0.0
    mantissa = s[:-2]
    exponent = s[-2:]
    return float(mantissa) * 10**int(exponent)


def calculate_orbital_period(mean_motion):
    """Вычисляет орбитальный период в минутах"""
    if mean_motion == 0:
        return 0
    return 1440.0 / mean_motion


def calculate_semi_major_axis(mean_motion):
    """Вычисляет большую полуось орбиты в км"""
    if mean_motion == 0:
        return 0
    MU_EARTH = 398600.4418
    n = mean_motion * 2 * np.pi / 86400.0
    a = (MU_EARTH / (n ** 2)) ** (1/3)
    return a


def calculate_apogee_perigee(semi_major_axis, eccentricity):
    """Вычисляет апогей и перигей в км"""
    EARTH_RADIUS = 6378.137
    apogee = semi_major_axis * (1 + eccentricity) - EARTH_RADIUS
    perigee = semi_major_axis * (1 - eccentricity) - EARTH_RADIUS
    return apogee, perigee


def is_geo_orbit(mean_motion, tolerance=0.05):
    """Проверяет, находится ли объект на GEO"""
    return abs(mean_motion - 1.0) <= tolerance


def classify_orbit(inclination, mean_motion, eccentricity):
    """Классифицирует тип орбиты"""
    semi_major_axis = calculate_semi_major_axis(mean_motion)
    altitude = semi_major_axis - 6378.137

    if eccentricity > 0.2:
        return "HEO (Highly Elliptical)"

    if is_geo_orbit(mean_motion) and inclination < 5 and eccentricity < 0.01:
        return "GEO (Geosynchronous)"

    if 2000 < altitude < 35786:
        return "MEO (Medium Earth Orbit)"

    if altitude < 2000:
        if inclination > 80:
            return "LEO Polar"
        elif inclination < 10:
            return "LEO Equatorial"
        else:
            return "LEO"

    return "Other"


def calculate_altitude_trend(tle_history):
    """Вычисляет тренд изменения высоты орбиты"""
    if len(tle_history) < 2:
        return {"trend": "insufficient_data", "rate": 0.0}

    altitudes = []
    times = []

    for tle in tle_history:
        a = calculate_semi_major_axis(tle['mean_motion'])
        altitude = a - 6378.137
        altitudes.append(altitude)
        times.append(tle['epoch']['datetime'])

    if len(altitudes) >= 2:
        time_deltas = [(t - times[0]).total_seconds() / 86400 for t in times]
        delta_alt = altitudes[-1] - altitudes[0]
        delta_time = time_deltas[-1]

        if delta_time > 0:
            rate = delta_alt / delta_time

            if rate < -0.5:
                trend = "rapidly_decaying"
            elif rate < -0.1:
                trend = "decaying"
            elif rate > 0.1:
                trend = "increasing"
            else:
                trend = "stable"

            return {
                "trend": trend,
                "rate": rate,
                "delta_altitude": delta_alt,
                "delta_days": delta_time
            }

    return {"trend": "stable", "rate": 0.0}


def estimate_reentry_time(perigee_km, bstar, mean_motion_dot):
    """Оценивает время до входа в атмосферу"""
    CRITICAL_ALTITUDE = 120

    if perigee_km > CRITICAL_ALTITUDE + 200:
        return {"status": "stable_orbit", "days_to_reentry": None, "confidence": "low"}

    if abs(bstar) > 0.0001 or abs(mean_motion_dot) > 0.001:
        altitude_margin = perigee_km - CRITICAL_ALTITUDE

        if altitude_margin < 50:
            days_estimate = max(1, altitude_margin * 2)
            confidence = "medium"
        elif altitude_margin < 100:
            days_estimate = altitude_margin * 5
            confidence = "low"
        else:
            days_estimate = altitude_margin * 10
            confidence = "very_low"

        return {"status": "decaying", "days_to_reentry": int(days_estimate), "confidence": confidence}

    return {"status": "stable_orbit", "days_to_reentry": None, "confidence": "low"}


if __name__ == "__main__":
    test_cases = [
        ("ISS (ZARYA)", "active satellite"),
        ("COSMOS 1408 DEB", "debris"),
        ("FALCON 9 R/B", "rocket body"),
        ("FENGYUN 1C DEB", "debris"),
        ("IRIDIUM 33 DEB", "debris fragment")
    ]

    for name, expected in test_cases:
        obj_type = classify_object_type(name)
        print(f"{name:30s} -> {obj_type.value:15s} (expected: {expected})")
