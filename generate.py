# generate.py
"""
Генерация детальных рекомендаций на основе TLE анализа
"""
import numpy as np
from parse_tle import (
    calculate_orbital_period,
    calculate_semi_major_axis,
    calculate_apogee_perigee,
    classify_orbit
)


def analyze_drift(raan, argument_perigee, mean_anomaly, epoch_data, previous_data=None):
    """
    Анализирует дрейф орбитальных параметров

    Returns:
        dict: информация о дрейфе
    """
    drift_info = {
        "raan_drift": "Нормальный",
        "perigee_drift": "Нормальный",
        "anomaly_drift": "Нормальный"
    }

    # Типичные диапазоны дрейфа для GEO
    # RAAN дрейфует примерно на 0.05-0.15 градусов/день
    # Argument of perigee более стабилен

    if previous_data:
        # Здесь можно сравнить с предыдущими данными
        pass

    return drift_info


def assess_station_keeping(inclination, eccentricity, mean_motion):
    """
    Оценивает необходимость коррекции орбиты (station-keeping)

    Returns:
        dict: рекомендации по коррекции
    """
    recommendations = []
    urgency = "Низкая"

    # Идеальные параметры для GEO
    IDEAL_INCLINATION = 0.0
    IDEAL_ECCENTRICITY = 0.0
    IDEAL_MEAN_MOTION = 1.0

    # Допуски
    INCLINATION_TOLERANCE = 0.1  # градусов
    ECCENTRICITY_TOLERANCE = 0.001
    MEAN_MOTION_TOLERANCE = 0.01

    # Проверка наклонения
    if abs(inclination - IDEAL_INCLINATION) > INCLINATION_TOLERANCE:
        deviation = abs(inclination - IDEAL_INCLINATION)
        recommendations.append(
            f"Наклонение отклонено от идеального на {deviation:.3f}°. "
            f"Рекомендуется коррекция North-South."
        )
        if deviation > 1.0:
            urgency = "Высокая"
        elif deviation > 0.5:
            urgency = "Средняя"

    # Проверка эксцентриситета
    if eccentricity > IDEAL_ECCENTRICITY + ECCENTRICITY_TOLERANCE:
        recommendations.append(
            f"Эксцентриситет повышен: {eccentricity:.6f}. "
            f"Рекомендуется коррекция East-West для циркуляризации орбиты."
        )
        if eccentricity > 0.01:
            urgency = "Высокая"

    # Проверка среднего движения (дрейф по долготе)
    if abs(mean_motion - IDEAL_MEAN_MOTION) > MEAN_MOTION_TOLERANCE:
        drift_rate = (mean_motion - IDEAL_MEAN_MOTION) * 360  # градусов/день
        recommendations.append(
            f"Дрейф по долготе: {drift_rate:.2f}°/день. "
            f"Требуется East-West коррекция."
        )
        if abs(drift_rate) > 0.5:
            urgency = "Средняя"

    if not recommendations:
        recommendations.append("Орбита стабильна. Коррекция не требуется в ближайшее время.")

    return {
        "urgency": urgency,
        "recommendations": recommendations
    }


def assess_collision_risk(altitude_km, eccentricity, inclination):
    """
    Оценивает риск столкновения на основе орбитальных параметров

    Returns:
        dict: информация о рисках
    """
    risk_level = "Низкий"
    warnings = []

    # GEO пояс - высокая плотность спутников
    if 35700 < altitude_km < 35900:
        warnings.append(
            "Объект в плотном GEO поясе. Рекомендуется мониторинг сближений."
        )
        risk_level = "Средний"

    # Высокий эксцентриситет - пересечение с другими поясами
    if eccentricity > 0.01:
        warnings.append(
            "Высокий эксцентриситет увеличивает риск пересечения с объектами на других орбитах."
        )
        risk_level = "Средний"

    if not warnings:
        warnings.append("Риск столкновения в пределах нормы.")

    return {
        "risk_level": risk_level,
        "warnings": warnings
    }


def calculate_fuel_budget(inclination_deviation, eccentricity, mean_motion_drift):
    """
    Оценивает потребность в топливе для коррекций

    Returns:
        dict: оценка расхода топлива
    """
    # Упрощенные коэффициенты (в реальности зависят от массы спутника)
    # Delta-V в м/с на градус наклонения: ~50 м/с
    # Delta-V для коррекции эксцентриситета: ~2-5 м/с на 0.001

    delta_v_inclination = abs(inclination_deviation) * 50  # м/с
    delta_v_eccentricity = eccentricity * 2000  # м/с
    delta_v_drift = abs(mean_motion_drift - 1.0) * 100  # м/с

    total_delta_v = delta_v_inclination + delta_v_eccentricity + delta_v_drift

    # Классификация
    if total_delta_v < 10:
        budget_status = "Минимальный расход"
    elif total_delta_v < 50:
        budget_status = "Умеренный расход"
    elif total_delta_v < 100:
        budget_status = "Значительный расход"
    else:
        budget_status = "Критический расход"

    return {
        "total_delta_v": total_delta_v,
        "budget_status": budget_status,
        "breakdown": {
            "inclination_correction": delta_v_inclination,
            "eccentricity_correction": delta_v_eccentricity,
            "drift_correction": delta_v_drift
        }
    }


def assess_operational_status(bstar, mean_motion_dot, age_days):
    """
    Оценивает операционный статус спутника

    Returns:
        dict: статус и предупреждения
    """
    status = "Активный"
    warnings = []

    # BSTAR - коэффициент атмосферного торможения
    if abs(bstar) > 0.0001:
        warnings.append(
            f"Повышенное атмосферное торможение (BSTAR={bstar:.2e}). "
            f"Возможно снижение орбиты."
        )
        status = "Требует внимания"

    # Производная среднего движения
    if abs(mean_motion_dot) > 0.0001:
        warnings.append(
            f"Значительное изменение орбитального периода (dn/dt={mean_motion_dot:.2e}). "
            f"Рекомендуется анализ причин."
        )

    # Возраст элементов
    if age_days > 7:
        warnings.append(
            f"Элементы устарели ({age_days:.1f} дней). "
            f"Рекомендуется обновление TLE."
        )
        status = "Устаревшие данные"

    if not warnings:
        warnings.append("Операционный статус: норма.")

    return {
        "status": status,
        "warnings": warnings
    }


def generate_detailed_text_with_values(tle_struct, reconstruction_error=None,
                                       anomaly_score=None, threshold=None):
    """
    Генерирует детальный анализ и рекомендации на основе TLE данных

    Args:
        tle_struct: структура с TLE данными
        reconstruction_error: ошибка реконструкции от модели (опционально)
        anomaly_score: оценка аномальности от модели (опционально)
        threshold: порог для определения аномалии (опционально)

    Returns:
        str: форматированный текст с рекомендациями
    """
    # Проверка наличия данных
    if all(tle_struct.get(k) is None for k in [
        "inclination", "raan", "eccentricity",
        "argument_perigee", "mean_anomaly", "mean_motion"
    ]):
        return "[Недостаточно данных для генерации рекомендаций]"

    # Извлечение параметров
    name = tle_struct.get("name", "Неизвестный объект")
    inclination = tle_struct.get("inclination", 0)
    raan = tle_struct.get("raan", 0)
    eccentricity = tle_struct.get("eccentricity", 0)
    argument_perigee = tle_struct.get("argument_perigee", 0)
    mean_anomaly = tle_struct.get("mean_anomaly", 0)
    mean_motion = tle_struct.get("mean_motion", 0)
    bstar = tle_struct.get("bstar", 0)
    mean_motion_dot = tle_struct.get("mean_motion_dot", 0)
    epoch = tle_struct.get("epoch", {})

    # Вычисления
    orbital_period = calculate_orbital_period(mean_motion)
    semi_major_axis = calculate_semi_major_axis(mean_motion)
    apogee, perigee = calculate_apogee_perigee(semi_major_axis, eccentricity)
    orbit_type = classify_orbit(inclination, mean_motion, eccentricity)

    # Возраст данных
    from datetime import datetime
    if 'datetime' in epoch:
        age_days = (datetime.now() - epoch['datetime']).total_seconds() / 86400
    else:
        age_days = 0

    # Анализы
    station_keeping = assess_station_keeping(inclination, eccentricity, mean_motion)
    collision_risk = assess_collision_risk(semi_major_axis - 6378.137, eccentricity, inclination)
    fuel_budget = calculate_fuel_budget(inclination, eccentricity, mean_motion)
    operational = assess_operational_status(bstar, mean_motion_dot, age_days)

    # Формирование текста
    lines = []
    lines.append("=" * 70)
    lines.append(f"АНАЛИЗ ОРБИТАЛЬНЫХ ПАРАМЕТРОВ: {name}")
    lines.append("=" * 70)

    # Базовые параметры
    lines.append("\n📊 ОРБИТАЛЬНЫЕ ПАРАМЕТРЫ:")
    lines.append(f"  • Тип орбиты: {orbit_type}")
    lines.append(f"  • Период обращения: {orbital_period:.2f} минут ({orbital_period / 60:.2f} часов)")
    lines.append(f"  • Большая полуось: {semi_major_axis:.1f} км")
    lines.append(f"  • Высота апогея: {apogee:.1f} км")
    lines.append(f"  • Высота перигея: {perigee:.1f} км")
    lines.append(f"  • Наклонение: {inclination:.4f}°")
    lines.append(f"  • RAAN: {raan:.4f}°")
    lines.append(f"  • Эксцентриситет: {eccentricity:.6f}")
    lines.append(f"  • Аргумент перигея: {argument_perigee:.4f}°")
    lines.append(f"  • Средняя аномалия: {mean_anomaly:.4f}°")
    lines.append(f"  • Среднее движение: {mean_motion:.8f} об/день")

    # Аномалии от модели
    if reconstruction_error is not None:
        lines.append(f"\n🤖 ОЦЕНКА МОДЕЛИ:")
        lines.append(f"  • Ошибка реконструкции: {reconstruction_error:.6f}")

        if anomaly_score is not None:
            lines.append(f"  • Оценка аномальности: {anomaly_score:.4f}")

        if threshold is not None:
            is_anomaly = reconstruction_error > threshold
            status_emoji = "⚠️" if is_anomaly else "✅"
            status_text = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly else "В пределах нормы"
            lines.append(f"  • Статус: {status_emoji} {status_text}")

    # Station-keeping
    lines.append(f"\n🛰️ КОРРЕКЦИЯ ОРБИТЫ (Station-Keeping):")
    lines.append(f"  • Срочность: {station_keeping['urgency']}")
    for rec in station_keeping['recommendations']:
        lines.append(f"  • {rec}")

    # Топливо
    lines.append(f"\n⛽ ОЦЕНКА РАСХОДА ТОПЛИВА:")
    lines.append(f"  • Общий Delta-V: {fuel_budget['total_delta_v']:.2f} м/с")
    lines.append(f"  • Статус: {fuel_budget['budget_status']}")
    lines.append(f"  • Коррекция наклонения: {fuel_budget['breakdown']['inclination_correction']:.2f} м/с")
    lines.append(f"  • Коррекция эксцентриситета: {fuel_budget['breakdown']['eccentricity_correction']:.2f} м/с")
    lines.append(f"  • Коррекция дрейфа: {fuel_budget['breakdown']['drift_correction']:.2f} м/с")

    # Риск столкновения
    lines.append(f"\n⚠️ РИСК СТОЛКНОВЕНИЯ:")
    lines.append(f"  • Уровень риска: {collision_risk['risk_level']}")
    for warning in collision_risk['warnings']:
        lines.append(f"  • {warning}")

    # Операционный статус
    lines.append(f"\n🔧 ОПЕРАЦИОННЫЙ СТАТУС:")
    lines.append(f"  • Статус: {operational['status']}")
    for warning in operational['warnings']:
        lines.append(f"  • {warning}")

    # Рекомендации
    lines.append(f"\n💡 ИТОГОВЫЕ РЕКОМЕНДАЦИИ:")

    priority_actions = []
    if station_keeping['urgency'] == "Высокая":
        priority_actions.append("🔴 СРОЧНО: Требуется немедленная коррекция орбиты")
    elif station_keeping['urgency'] == "Средняя":
        priority_actions.append("🟡 ВНИМАНИЕ: Запланировать коррекцию орбиты")

    if collision_risk['risk_level'] in ["Высокий", "Средний"]:
        priority_actions.append("⚠️ Усилить мониторинг сближений с другими объектами")

    if fuel_budget['budget_status'] == "Критический расход":
        priority_actions.append("⛽ КРИТИЧНО: Оптимизировать стратегию расхода топлива")

    if operational['status'] != "Активный":
        priority_actions.append("🔧 Проверить операционный статус спутника")

    if priority_actions:
        for action in priority_actions:
            lines.append(f"  {action}")
    else:
        lines.append("  ✅ Спутник в стабильном состоянии")
        lines.append("  📋 Продолжить регулярный мониторинг")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def generate_summary_text(tle_struct):
    """
    Генерирует краткий summary для быстрого просмотра

    Returns:
        str: краткое описание
    """
    name = tle_struct.get("name", "Неизвестный")
    mean_motion = tle_struct.get("mean_motion", 0)
    inclination = tle_struct.get("inclination", 0)
    eccentricity = tle_struct.get("eccentricity", 0)

    orbit_type = classify_orbit(inclination, mean_motion, eccentricity)
    period = calculate_orbital_period(mean_motion)

    summary = (
        f"{name} | {orbit_type} | "
        f"Период: {period:.1f}мин | "
        f"Накл: {inclination:.2f}° | "
        f"Эксц: {eccentricity:.4f}"
    )

    return summary


if __name__ == "__main__":
    # Тестовый пример
    test_tle = {
        "name": "TEST SATELLITE",
        "inclination": 0.05,
        "raan": 45.0,
        "eccentricity": 0.0002,
        "argument_perigee": 90.0,
        "mean_anomaly": 180.0,
        "mean_motion": 1.00273,
        "bstar": 0.00001,
        "mean_motion_dot": 0.00000001,
        "epoch": {"year": 2024, "day_of_year": 1}
    }

    text = generate_detailed_text_with_values(test_tle, reconstruction_error=0.0012)
    print(text)
