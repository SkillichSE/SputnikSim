# parse_tle.py
"""
Парсер TLE (Two-Line Element) данных
Извлекает орбитальные параметры спутника из формата TLE
"""
import numpy as np
from datetime import datetime, timedelta


class TLEParseError(Exception):
    """Исключение для ошибок парсинга TLE"""
    pass


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

        # Номер каталога
        catalog_number = line1[2:7].strip()

        # Классификация
        classification = line1[7]

        # Международное обозначение
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
            "line2": line2.strip()
        }

        return vector, tle_struct

    except (ValueError, IndexError) as e:
        raise TLEParseError(f"Ошибка парсинга TLE: {str(e)}")


def parse_scientific_notation(s):
    """
    Парсит научную нотацию TLE формата (без 'E')
    Пример: '-12345-3' = -0.12345e-3
    """
    if not s or s.isspace():
        return 0.0

    s = s.strip()

    # Проверяем знак в конце (экспонента)
    if s[-2] in ['+', '-']:
        mantissa = s[:-2]
        exponent = s[-2:]
        return float(mantissa) * (10 ** int(exponent))
    else:
        return float(s)


def calculate_orbital_period(mean_motion):
    """
    Вычисляет орбитальный период в минутах

    Args:
        mean_motion: среднее движение в оборотах/день

    Returns:
        float: период в минутах
    """
    if mean_motion == 0:
        return 0
    return 1440.0 / mean_motion  # 1440 минут в дне


def calculate_semi_major_axis(mean_motion):
    """
    Вычисляет большую полуось орбиты в км
    Использует третий закон Кеплера

    Args:
        mean_motion: среднее движение в оборотах/день

    Returns:
        float: большая полуось в км
    """
    if mean_motion == 0:
        return 0

    # Константы
    MU_EARTH = 398600.4418  # км³/с² (гравитационный параметр Земли)

    # Конвертация mean_motion из оборотов/день в рад/сек
    n = mean_motion * 2 * np.pi / 86400.0  # рад/сек

    # a³ = μ/n²
    a = (MU_EARTH / (n ** 2)) ** (1 / 3)

    return a


def calculate_apogee_perigee(semi_major_axis, eccentricity):
    """
    Вычисляет апогей и перигей в км (от поверхности Земли)

    Args:
        semi_major_axis: большая полуось в км
        eccentricity: эксцентриситет

    Returns:
        tuple: (apogee, perigee) в км от поверхности
    """
    EARTH_RADIUS = 6378.137  # км

    apogee = semi_major_axis * (1 + eccentricity) - EARTH_RADIUS
    perigee = semi_major_axis * (1 - eccentricity) - EARTH_RADIUS

    return apogee, perigee


def is_geo_orbit(mean_motion, tolerance=0.05):
    """
    Проверяет, находится ли объект на геостационарной орбите

    Args:
        mean_motion: среднее движение в оборотах/день
        tolerance: допустимое отклонение от 1.0 оборота/день

    Returns:
        bool: True если GEO
    """
    return abs(mean_motion - 1.0) <= tolerance


def classify_orbit(inclination, mean_motion, eccentricity):
    """
    Классифицирует тип орбиты

    Returns:
        str: тип орбиты
    """
    semi_major_axis = calculate_semi_major_axis(mean_motion)
    altitude = semi_major_axis - 6378.137  # примерная высота

    # GEO
    if is_geo_orbit(mean_motion) and inclination < 5 and eccentricity < 0.01:
        return "GEO (Geosynchronous)"

    # MEO
    if 2000 < altitude < 35786:
        return "MEO (Medium Earth Orbit)"

    # LEO
    if altitude < 2000:
        if inclination > 80:
            return "LEO Polar"
        elif inclination < 10:
            return "LEO Equatorial"
        else:
            return "LEO"

    # HEO
    if eccentricity > 0.2:
        return "HEO (Highly Elliptical)"

    return "Other"


if __name__ == "__main__":
    # Тестовый пример
    test_name = "ISS (ZARYA)"
    test_line1 = "1 25544U 98067A   24001.50000000  .00012345  00000-0  12345-3 0  9992"
    test_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391111111"

    try:
        vector, tle_data = parse_tle(test_name, test_line1, test_line2)
        print("Вектор:", vector)
        print("\nТип орбиты:", classify_orbit(
            tle_data['inclination'],
            tle_data['mean_motion'],
            tle_data['eccentricity']
        ))
        print("\nПериод орбиты:", f"{calculate_orbital_period(tle_data['mean_motion']):.2f} мин")
    except TLEParseError as e:
        print(f"Ошибка: {e}")
