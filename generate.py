"""
Генерация детальных контекстно-зависимых рекомендаций на основе TLE анализа
Учитывает тип объекта (спутник/мусор) и исторические данные
"""
import numpy as np
from datetime import datetime
from parse_tle import (
    calculate_orbital_period,
    calculate_semi_major_axis,
    calculate_apogee_perigee,
    classify_orbit,
    ObjectType,
    OperationalStatus,
    calculate_altitude_trend,
    estimate_reentry_time
)


def calculate_confidence_score(reconstruction_error, bstar, mean_motion_dot, tle_age_days):
    """
    Вычисляет доверительный интервал для детекции аномалий
    Учитывает погрешность измерений и качество данных

    Returns:
        dict: {confidence_percent, quality_factors, reliability}
    """
    confidence = 100.0  # Начинаем со 100%
    factors = []

    # Фактор 1: Возраст TLE
    if tle_age_days > 7:
        penalty = min(30, (tle_age_days - 7) * 3)
        confidence -= penalty
        factors.append(f"Устаревшие данные (-{penalty:.0f}%)")

    # Фактор 2: BSTAR (погрешность модели атмосферы)
    if abs(bstar) > 0.001:
        confidence -= 15
        factors.append("Высокое атмосферное торможение (-15%)")
    elif abs(bstar) > 0.0001:
        confidence -= 10
        factors.append("Умеренное атмосферное торможение (-10%)")

    # Фактор 3: Производная среднего движения
    if abs(mean_motion_dot) > 0.001:
        confidence -= 10
        factors.append("Нестабильность орбиты (-10%)")

    # Фактор 4: Величина ошибки реконструкции
    if reconstruction_error is not None:
        if reconstruction_error > 0.1:
            confidence -= 20
            factors.append("Очень высокая ошибка реконструкции (-20%)")
        elif reconstruction_error > 0.05:
            confidence -= 10
            factors.append("Высокая ошибка реконструкции (-10%)")

    confidence = max(0, min(100, confidence))

    # Определение надежности
    if confidence >= 80:
        reliability = "Высокая"
    elif confidence >= 60:
        reliability = "Средняя"
    elif confidence >= 40:
        reliability = "Низкая"
    else:
        reliability = "Очень низкая"

    return {
        "confidence_percent": round(confidence, 1),
        "quality_factors": factors if factors else ["Нет значимых факторов"],
        "reliability": reliability
    }


def analyze_reconstruction_error_components(original_vector, reconstructed_vector):
    """
    Анализирует компоненты ошибки реконструкции и определяет основную причину аномалии

    Args:
        original_vector: оригинальный вектор [inclination, raan, ecc, arg_perigee, mean_anomaly, mean_motion]
        reconstructed_vector: восстановленный вектор

    Returns:
        dict: детальный анализ ошибок по каждому параметру
    """
    param_names = [
        "Наклонение (Inclination)",
        "RAAN",
        "Эксцентриситет",
        "Аргумент перигея",
        "Средняя аномалия",
        "Среднее движение"
    ]

    param_units = ["°", "°", "", "°", "°", "об/день"]

    # Относительные веса важности (для приоритизации)
    param_weights = [2.0, 1.5, 3.0, 1.0, 0.5, 2.5]  # Эксцентриситет и mean_motion важнее

    errors = []
    for i, (orig, recon, name, unit, weight) in enumerate(zip(
        original_vector, reconstructed_vector, param_names, param_units, param_weights
    )):
        abs_error = abs(orig - recon)
        rel_error = (abs_error / (abs(orig) + 1e-10)) * 100  # процентная ошибка
        weighted_error = abs_error * weight

        errors.append({
            "index": i,
            "parameter": name,
            "original": orig,
            "reconstructed": recon,
            "absolute_error": abs_error,
            "relative_error": rel_error,
            "weighted_error": weighted_error,
            "unit": unit
        })

    # Сортируем по взвешенной ошибке
    errors_sorted = sorted(errors, key=lambda x: x['weighted_error'], reverse=True)

    # Определяем основную причину аномалии
    main_cause = errors_sorted[0]

    # Формируем диагностическое сообщение
    if main_cause['weighted_error'] > 0.01:
        if "Среднее движение" in main_cause['parameter']:
            diagnosis = (
                f"Аномалия вызвана скачком среднего движения: "
                f"ожидалось {main_cause['reconstructed']:.6f}, "
                f"наблюдается {main_cause['original']:.6f} об/день "
                f"(отклонение: {main_cause['absolute_error']:.6f})"
            )
        elif "Эксцентриситет" in main_cause['parameter']:
            diagnosis = (
                f"Аномалия вызвана изменением эксцентриситета: "
                f"отклонение {main_cause['absolute_error']:.6f} "
                f"({main_cause['relative_error']:.1f}% от нормы)"
            )
        elif "Наклонение" in main_cause['parameter']:
            diagnosis = (
                f"Аномалия вызвана изменением наклонения: "
                f"отклонение {main_cause['absolute_error']:.4f}° "
                f"({main_cause['relative_error']:.1f}% от нормы)"
            )
        elif "RAAN" in main_cause['parameter']:
            diagnosis = (
                f"Аномалия вызвана смещением RAAN: "
                f"отклонение {main_cause['absolute_error']:.4f}°"
            )
        else:
            diagnosis = (
                f"Аномалия в параметре '{main_cause['parameter']}': "
                f"отклонение {main_cause['absolute_error']:.4f}{main_cause['unit']}"
            )
    else:
        diagnosis = "Аномалия распределена по нескольким параметрам (комплексное нарушение)"

    return {
        "components": errors_sorted,
        "main_cause": main_cause,
        "diagnosis": diagnosis,
        "top_3_contributors": errors_sorted[:3]
    }


def calculate_correction_angles(tle_struct, target_params=None):
    """
    Рассчитывает необходимые углы и величины коррекции орбиты

    Args:
        tle_struct: структура TLE
        target_params: целевые параметры (если None, используются идеальные для GEO)

    Returns:
        dict: требуемые коррекции в градусах и м/с
    """
    # Текущие параметры
    inclination = tle_struct.get('inclination', 0)
    raan = tle_struct.get('raan', 0)
    eccentricity = tle_struct.get('eccentricity', 0)
    argument_perigee = tle_struct.get('argument_perigee', 0)
    mean_motion = tle_struct.get('mean_motion', 0)

    # Целевые параметры (идеальная GEO по умолчанию)
    if target_params is None:
        target_params = {
            'inclination': 0.0,
            'eccentricity': 0.0,
            'mean_motion': 1.0
        }

    # === 1. КОРРЕКЦИЯ НАКЛОНЕНИЯ (North-South) ===
    delta_inclination = target_params['inclination'] - inclination

    # Delta-V для изменения наклонения (на экваторе)
    # ΔV = 2 * V_орб * sin(Δi/2)
    semi_major_axis = calculate_semi_major_axis(mean_motion)
    orbital_velocity = np.sqrt(398600.4418 / semi_major_axis)  # км/с
    delta_v_inclination = 2 * orbital_velocity * np.sin(np.radians(abs(delta_inclination) / 2)) * 1000  # м/с

    # === 2. КОРРЕКЦИЯ ЭКСЦЕНТРИСИТЕТА ===
    delta_eccentricity = target_params['eccentricity'] - eccentricity

    # Упрощенная формула для малых эксцентриситетов
    # ΔV ≈ V_орб * Δe
    delta_v_eccentricity = orbital_velocity * abs(delta_eccentricity) * 1000  # м/с

    # === 3. КОРРЕКЦИЯ ДРЕЙФА (East-West) ===
    delta_mean_motion = target_params['mean_motion'] - mean_motion

    # Для GEO: изменение периода на 1 секунду ≈ дрейф 0.25°/день
    # ΔV для изменения периода
    period_current = 1440.0 / mean_motion  # минуты
    period_target = 1440.0 / target_params['mean_motion']
    delta_period = period_target - period_current  # минуты

    # ΔV ≈ (3/2) * V_орб * (ΔT/T)
    delta_v_drift = (3/2) * orbital_velocity * (delta_period * 60 / (period_current * 60)) * 1000  # м/с

    # === 4. ОПТИМАЛЬНЫЙ МОМЕНТ КОРРЕКЦИИ ===
    # Для коррекции наклонения лучше всего на экваторе (True Anomaly = 0° или 180°)
    # Для коррекции эксцентриситета - в апогее или перигее

    if abs(delta_inclination) > 0.1:
        optimal_timing = "На экваторе (True Anomaly = 0° или 180°)"
        burn_location = "Восходящий или нисходящий узел"
    elif abs(delta_eccentricity) > 0.001:
        optimal_timing = "В апогее (для уменьшения эксцентриситета)"
        burn_location = "Апогей орбиты"
    else:
        optimal_timing = "В любой точке орбиты"
        burn_location = "Не критично"

    corrections = {
        "inclination": {
            "current": inclination,
            "target": target_params['inclination'],
            "delta_degrees": delta_inclination,
            "delta_v_ms": delta_v_inclination,
            "direction": "North" if delta_inclination < 0 else "South"
        },
        "eccentricity": {
            "current": eccentricity,
            "target": target_params['eccentricity'],
            "delta": delta_eccentricity,
            "delta_v_ms": delta_v_eccentricity
        },
        "drift": {
            "current_mean_motion": mean_motion,
            "target_mean_motion": target_params['mean_motion'],
            "delta_degrees_per_day": delta_mean_motion * 360,
            "delta_v_ms": delta_v_drift,
            "direction": "East" if delta_mean_motion > 0 else "West"
        },
        "total": {
            "delta_v_ms": delta_v_inclination + delta_v_eccentricity + abs(delta_v_drift),
            "optimal_timing": optimal_timing,
            "burn_location": burn_location
        }
    }

    return corrections


def determine_satellite_status(tle_struct, object_type, reconstruction_error, threshold):
    """
    Определяет статус спутника на основе всех доступных данных

    Returns:
        dict: {status, confidence, reasoning}
    """
    bstar = tle_struct.get('bstar', 0)
    mean_motion_dot = tle_struct.get('mean_motion_dot', 0)
    status_by_name = tle_struct.get('status_by_name', OperationalStatus.UNKNOWN)

    age_days = 0
    if 'epoch' in tle_struct and 'datetime' in tle_struct['epoch']:
        age_days = (datetime.now() - tle_struct['epoch']['datetime']).total_seconds() / 86400

    is_debris = object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]
    is_anomaly = reconstruction_error > threshold if reconstruction_error else False

    reasoning = []

    # === ОПРЕДЕЛЕНИЕ СТАТУСА ===

    if is_debris:
        status = "Неуправляемый объект (Космический мусор)"
        confidence = 100
        reasoning.append("Объект классифицирован как космический мусор")

    elif status_by_name in [OperationalStatus.DEAD, OperationalStatus.DECAYED, OperationalStatus.FRAGMENTED]:
        status = f"Выведен из строя ({status_by_name.value.upper()})"
        confidence = 95
        reasoning.append(f"Статус в базе: {status_by_name.value}")

    elif is_anomaly:
        # Аномалия может означать либо маневр, либо проблему
        if abs(mean_motion_dot) > 0.0005:
            status = "Вероятный активный маневр"
            confidence = 70
            reasoning.append(f"Обнаружена аномалия (ошибка: {reconstruction_error:.4f})")
            reasoning.append(f"Высокая производная среднего движения: {mean_motion_dot:.2e}")
        else:
            status = "Возможна потеря управления"
            confidence = 60
            reasoning.append(f"Обнаружена аномалия без признаков маневра")

    elif abs(bstar) > 0.0001 or abs(mean_motion_dot) > 0.0001:
        status = "Функционирующий (требует внимания)"
        confidence = 75
        reasoning.append("Орбита нестабильна, возможны коррекции")

    elif age_days > 7:
        status = "Статус неопределен (данные устарели)"
        confidence = 30
        reasoning.append(f"Данные устарели ({age_days:.1f} дней)")

    else:
        status = "Функционирующий"
        confidence = 90
        reasoning.append("Все параметры в норме")

    return {
        "status": status,
        "confidence": confidence,
        "reasoning": reasoning
    }


def assess_station_keeping_for_satellite(inclination, eccentricity, mean_motion):
    """Оценка station-keeping для АКТИВНЫХ спутников"""
    recommendations = []
    urgency = "Низкая"

    IDEAL_INCLINATION = 0.0
    IDEAL_ECCENTRICITY = 0.0
    IDEAL_MEAN_MOTION = 1.0

    INCLINATION_TOLERANCE = 0.1
    ECCENTRICITY_TOLERANCE = 0.001
    MEAN_MOTION_TOLERANCE = 0.01

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

    if eccentricity > IDEAL_ECCENTRICITY + ECCENTRICITY_TOLERANCE:
        recommendations.append(
            f"Эксцентриситет повышен: {eccentricity:.6f}. "
            f"Рекомендуется коррекция East-West для циркуляризации орбиты."
        )
        if eccentricity > 0.01:
            urgency = "Высокая"

    if abs(mean_motion - IDEAL_MEAN_MOTION) > MEAN_MOTION_TOLERANCE:
        drift_rate = (mean_motion - IDEAL_MEAN_MOTION) * 360
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
        "recommendations": recommendations,
        "applicable": True
    }


def assess_debris_degradation(tle_struct, tle_history=None):
    """Анализ деградации орбиты для космического МУСОРА"""
    recommendations = []
    urgency = "Мониторинг"

    semi_major_axis = calculate_semi_major_axis(tle_struct['mean_motion'])
    apogee, perigee = calculate_apogee_perigee(semi_major_axis, tle_struct['eccentricity'])

    # Тренд высоты
    if tle_history and len(tle_history) > 1:
        trend_info = calculate_altitude_trend(tle_history)

        if trend_info['trend'] == "rapidly_decaying":
            recommendations.append(
                f"⚠️ КРИТИЧНО: Быстрое снижение орбиты ({trend_info['rate']:.2f} км/день). "
                f"Объект потерял {abs(trend_info['delta_altitude']):.1f} км за {trend_info['delta_days']:.0f} дней."
            )
            urgency = "Критический"

            # Оценка времени до входа
            reentry = estimate_reentry_time(perigee, tle_struct['bstar'], tle_struct['mean_motion_dot'])
            if reentry['status'] == 'decaying' and reentry['days_to_reentry']:
                recommendations.append(
                    f"📅 Прогноз входа в атмосферу: ~{reentry['days_to_reentry']} дней "
                    f"(уверенность: {reentry['confidence']})"
                )

        elif trend_info['trend'] == "decaying":
            recommendations.append(
                f"Постепенное снижение орбиты ({trend_info['rate']:.3f} км/день). "
                f"Продолжить мониторинг."
            )
            urgency = "Повышенный"

    # Проверка критической высоты
    if perigee < 300:
        recommendations.append(
            f"⚠️ Низкий перигей ({perigee:.1f} км). "
            f"Повышенное атмосферное торможение."
        )
        urgency = "Повышенный"

    # BSTAR анализ
    if abs(tle_struct['bstar']) > 0.0001:
        recommendations.append(
            f"Высокий BSTAR коэффициент ({tle_struct['bstar']:.2e}). "
            f"Активное торможение в атмосфере."
        )

    if not recommendations:
        recommendations.append("Орбита стабильна. Деградация не обнаружена.")

    return {
        "urgency": urgency,
        "recommendations": recommendations,
        "applicable": True
    }


def assess_conjunction_risk(tle_struct, object_type):
    """Оценка риска опасного сближения (Conjunction Assessment)"""
    warnings = []
    risk_level = "Низкий"

    semi_major_axis = calculate_semi_major_axis(tle_struct['mean_motion'])
    altitude = semi_major_axis - 6378.137

    # Для мусора риск выше
    if object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]:
        risk_multiplier = 1.5
        warnings.append("Объект классифицирован как мусор - повышенный риск столкновения")
    else:
        risk_multiplier = 1.0

    # GEO пояс
    if 35700 < altitude < 35900:
        warnings.append("Объект в плотном GEO поясе (±100 км от 35786 км)")
        risk_level = "Средний"

    # LEO пояс
    if 400 < altitude < 600:
        warnings.append("Объект в плотном LEO поясе (400-600 км)")
        risk_level = "Средний"

    # Высокий эксцентриситет
    if tle_struct['eccentricity'] > 0.01:
        warnings.append(
            f"Высокий эксцентриситет ({tle_struct['eccentricity']:.4f}) - "
            f"пересечение множества орбитальных слоев"
        )
        risk_level = "Повышенный"

    # Специфично для мусора
    if object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT]:
        warnings.append(
            "🔴 РЕКОМЕНДАЦИЯ: Включить объект в каталог для Conjunction Data Messages (CDM)"
        )
        if risk_level == "Средний":
            risk_level = "Высокий"

    if not warnings:
        warnings.append("Риск столкновения в пределах нормы")

    return {
        "risk_level": risk_level,
        "warnings": warnings
    }


def calculate_fuel_budget(inclination_deviation, eccentricity, mean_motion_drift):
    """Расчет расхода топлива (только для спутников)"""
    delta_v_inclination = abs(inclination_deviation) * 50
    delta_v_eccentricity = eccentricity * 2000
    delta_v_drift = abs(mean_motion_drift - 1.0) * 100

    total_delta_v = delta_v_inclination + delta_v_eccentricity + delta_v_drift

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


def assess_operational_status(tle_struct, object_type):
    """Оценка операционного статуса с учетом типа объекта"""
    status = "Активный"
    warnings = []

    bstar = tle_struct['bstar']
    mean_motion_dot = tle_struct['mean_motion_dot']
    age_days = (datetime.now() - tle_struct['epoch']['datetime']).total_seconds() / 86400

    # Для мусора другие критерии
    if object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]:
        status = "Неуправляемый объект"
        warnings.append("Статус: космический мусор (управление невозможно)")
    else:
        # Для спутников
        if abs(bstar) > 0.0001:
            warnings.append(
                f"Повышенное атмосферное торможение (BSTAR={bstar:.2e}). "
                f"Возможно снижение орбиты."
            )
            status = "Требует внимания"

        if abs(mean_motion_dot) > 0.0001:
            warnings.append(
                f"Значительное изменение орбитального периода (dn/dt={mean_motion_dot:.2e}). "
                f"Рекомендуется анализ причин."
            )

    if age_days > 7:
        warnings.append(
            f"Элементы устарели ({age_days:.1f} дней). "
            f"Рекомендуется обновление TLE."
        )
        status = "Устаревшие данные"

    if not warnings:
        warnings.append("Операционный статус: норма")

    return {
        "status": status,
        "warnings": warnings
    }


def generate_context_aware_recommendations(tle_struct, reconstruction_error, anomaly_score,
                                          threshold, tle_history=None):
    """
    Генерирует контекстно-зависимые рекомендации

    Args:
        tle_struct: структура TLE
        reconstruction_error: ошибка реконструкции модели
        anomaly_score: оценка аномальности
        threshold: порог аномалии
        tle_history: история TLE для трендового анализа

    Returns:
        list: список приоритезированных рекомендаций
    """
    from datetime import datetime

    recommendations = []
    object_type = tle_struct.get('object_type', ObjectType.UNKNOWN)
    is_debris = object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]
    is_anomaly = reconstruction_error > threshold if reconstruction_error else False

    # === 1. КРИТИЧЕСКИЕ РЕКОМЕНДАЦИИ ===

    if is_anomaly:
        if is_debris:
            recommendations.append({
                "priority": "🔴 КРИТИЧНО",
                "action": "АНОМАЛИЯ ОБНАРУЖЕНА В КОСМИЧЕСКОМ МУСОРЕ",
                "details": [
                    f"Ошибка реконструкции: {reconstruction_error:.6f} (порог: {threshold:.6f})",
                    "Возможные причины: столкновение, фрагментация, резкое изменение орбиты",
                    "🚨 ДЕЙСТВИЯ: Немедленно проверить на предмет фрагментации",
                    "🚨 ДЕЙСТВИЯ: Усилить мониторинг сближений (Conjunction Assessment)",
                    "🚨 ДЕЙСТВИЯ: Проверить каталог на появление новых обломков"
                ]
            })
        else:
            recommendations.append({
                "priority": "🔴 КРИТИЧНО",
                "action": "АНОМАЛИЯ ОБНАРУЖЕНА В АКТИВНОМ СПУТНИКЕ",
                "details": [
                    f"Ошибка реконструкции: {reconstruction_error:.6f} (порог: {threshold:.6f})",
                    "Возможные причины: нештатный маневр, столкновение, потеря ориентации",
                    "🚨 ДЕЙСТВИЯ: Связаться с оператором спутника",
                    "🚨 ДЕЙСТВИЯ: Проверить телеметрию и журналы маневров",
                    "🚨 ДЕЙСТВИЯ: Оценить необходимость экстренной коррекции"
                ]
            })

    # === 2. АНАЛИЗ ДЕГРАДАЦИИ ДЛЯ МУСОРА ===

    if is_debris:
        degradation = assess_debris_degradation(tle_struct, tle_history)
        if degradation['urgency'] in ['Критический', 'Повышенный']:
            recommendations.append({
                "priority": f"⚠️ {degradation['urgency'].upper()}",
                "action": "АНАЛИЗ ДЕГРАДАЦИИ ОРБИТЫ",
                "details": degradation['recommendations']
            })

    # === 3. STATION-KEEPING ДЛЯ СПУТНИКОВ ===

    if not is_debris:
        station_keeping = assess_station_keeping_for_satellite(
            tle_struct['inclination'],
            tle_struct['eccentricity'],
            tle_struct['mean_motion']
        )
        if station_keeping['urgency'] in ['Высокая', 'Средняя']:
            recommendations.append({
                "priority": f"🟡 {station_keeping['urgency'].upper()}",
                "action": "КОРРЕКЦИЯ ОРБИТЫ (Station-Keeping)",
                "details": station_keeping['recommendations']
            })

    # === 4. РИСК СТОЛКНОВЕНИЯ ===

    conjunction = assess_conjunction_risk(tle_struct, object_type)
    if conjunction['risk_level'] in ['Высокий', 'Повышенный']:
        recommendations.append({
            "priority": "🟠 ПОВЫШЕННЫЙ РИСК",
            "action": "ОПАСНОЕ СБЛИЖЕНИЕ (Conjunction Assessment)",
            "details": conjunction['warnings']
        })

    # === 5. ТРЕНДОВЫЙ АНАЛИЗ ===

    if tle_history and len(tle_history) >= 3:
        trend = calculate_altitude_trend(tle_history)
        if trend['trend'] in ['rapidly_decaying', 'decaying']:
            recommendations.append({
                "priority": "📊 ТРЕНД",
                "action": "ОБНАРУЖЕНО СНИЖЕНИЕ ОРБИТЫ",
                "details": [
                    f"Скорость изменения: {trend['rate']:.3f} км/день",
                    f"Изменение за период: {trend['delta_altitude']:.1f} км за {trend['delta_days']:.0f} дней",
                    "Рекомендуется продолжить мониторинг"
                ]
            })

    # === 6. БАЗОВЫЕ РЕКОМЕНДАЦИИ ===

    if not recommendations:
        if is_debris:
            recommendations.append({
                "priority": "✅ НОРМА",
                "action": "Космический мусор в стабильном состоянии",
                "details": [
                    "Продолжить регулярный мониторинг",
                    "Обновлять TLE каждые 3-7 дней",
                    "Контролировать риски сближения"
                ]
            })
        else:
            recommendations.append({
                "priority": "✅ НОРМА",
                "action": "Спутник в стабильном состоянии",
                "details": [
                    "Продолжить регулярный мониторинг",
                    "Station-keeping не требуется в ближайшее время",
                    "Проверять TLE обновления каждые 1-3 дня"
                ]
            })

    return recommendations


def generate_detailed_text_with_values(tle_struct, reconstruction_error=None,
                                      anomaly_score=None, threshold=None,
                                      tle_history=None, reconstructed_vector=None):
    """
    Генерирует детальный отчет с учетом типа объекта и контекста

    Args:
        reconstructed_vector: восстановленный вектор от модели (для анализа ошибок)
    """
    name = tle_struct.get("name", "Неизвестный объект")
    object_type = tle_struct.get("object_type", ObjectType.UNKNOWN)
    status_by_name = tle_struct.get("status_by_name", OperationalStatus.UNKNOWN)

    is_debris = object_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]

    # Извлечение параметров
    inclination = tle_struct.get("inclination", 0)
    raan = tle_struct.get("raan", 0)
    eccentricity = tle_struct.get("eccentricity", 0)
    argument_perigee = tle_struct.get("argument_perigee", 0)
    mean_anomaly = tle_struct.get("mean_anomaly", 0)
    mean_motion = tle_struct.get("mean_motion", 0)
    bstar = tle_struct.get("bstar", 0)
    mean_motion_dot = tle_struct.get("mean_motion_dot", 0)
    epoch = tle_struct.get("epoch", {})
    catalog_number = tle_struct.get("catalog_number", "N/A")
    intl_designator = tle_struct.get("intl_designator", "N/A")

    # Вычисления
    orbital_period = calculate_orbital_period(mean_motion)
    semi_major_axis = calculate_semi_major_axis(mean_motion)
    apogee, perigee = calculate_apogee_perigee(semi_major_axis, eccentricity)
    orbit_type = classify_orbit(inclination, mean_motion, eccentricity)

    # Возраст данных
    if 'datetime' in epoch:
        age_days = (datetime.now() - epoch['datetime']).total_seconds() / 86400
    else:
        age_days = 0

    # Доверительный интервал
    confidence = calculate_confidence_score(reconstruction_error, bstar, mean_motion_dot, age_days)

    # Статус спутника
    satellite_status = determine_satellite_status(tle_struct, object_type, reconstruction_error, threshold)

    # Анализ компонент ошибки
    error_analysis = None
    if reconstructed_vector is not None and reconstruction_error is not None:
        original_vector = [inclination, raan, eccentricity, argument_perigee, mean_anomaly, mean_motion]
        error_analysis = analyze_reconstruction_error_components(original_vector, reconstructed_vector)

    # Расчет коррекции (только для спутников)
    corrections = None
    if not is_debris and object_type == ObjectType.SATELLITE:
        corrections = calculate_correction_angles(tle_struct)

    # Анализы
    operational = assess_operational_status(tle_struct, object_type)
    conjunction = assess_conjunction_risk(tle_struct, object_type)

    # Контекстные рекомендации
    context_recommendations = generate_context_aware_recommendations(
        tle_struct, reconstruction_error, anomaly_score, threshold, tle_history
    )

    # === Формирование текста ===

    lines = []
    lines.append("=" * 70)
    lines.append(f"АНАЛИЗ ОРБИТАЛЬНЫХ ПАРАМЕТРОВ: {name}")
    lines.append("=" * 70)

    # Классификация объекта
    lines.append(f"\n🏷️ КЛАССИФИКАЦИЯ ОБЪЕКТА:")
    lines.append(f"  • Тип: {object_type.value.upper().replace('_', ' ')}")
    lines.append(f"  • NORAD ID: {catalog_number}")
    lines.append(f"  • COSPAR ID: {intl_designator}")
    if status_by_name != OperationalStatus.UNKNOWN:
        lines.append(f"  • Статус по базе: {status_by_name.value.upper()}")

    # СТАТУС СПУТНИКА
    lines.append(f"\n🛰️ СТАТУС ОБЪЕКТА:")
    lines.append(f"  • Статус: {satellite_status['status']}")
    lines.append(f"  • Уверенность: {satellite_status['confidence']}%")
    lines.append(f"  • Обоснование:")
    for reason in satellite_status['reasoning']:
        lines.append(f"    - {reason}")

    # Базовые параметры
    lines.append(f"\n📊 ОРБИТАЛЬНЫЕ ПАРАМЕТРЫ:")
    lines.append(f"  • Тип орбиты: {orbit_type}")
    lines.append(f"  • Период обращения: {orbital_period:.2f} минут ({orbital_period/60:.2f} часов)")
    lines.append(f"  • Большая полуось: {semi_major_axis:.1f} км")
    lines.append(f"  • Высота апогея: {apogee:.1f} км")
    lines.append(f"  • Высота перигея: {perigee:.1f} км")
    lines.append(f"  • Наклонение: {inclination:.4f}°")
    lines.append(f"  • RAAN: {raan:.4f}°")
    lines.append(f"  • Эксцентриситет: {eccentricity:.6f}")
    lines.append(f"  • Среднее движение: {mean_motion:.8f} об/день")
    lines.append(f"  • Возраст данных: {age_days:.1f} дней")

    # Оценка модели
    if reconstruction_error is not None:
        lines.append(f"\n🤖 ОЦЕНКА МОДЕЛИ И ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ:")
        lines.append(f"  • Ошибка реконструкции: {reconstruction_error:.6f}")
        lines.append(f"  • Уверенность модели: {confidence['confidence_percent']:.1f}%")
        lines.append(f"  • Надежность: {confidence['reliability']}")

        if anomaly_score is not None:
            lines.append(f"  • Оценка аномальности: {anomaly_score:.4f}")

        if threshold is not None:
            is_anomaly = reconstruction_error > threshold
            status_emoji = "⚠️" if is_anomaly else "✅"
            status_text = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly else "В пределах нормы"
            lines.append(f"  • Статус: {status_emoji} {status_text}")

        lines.append(f"  • Факторы качества:")
        for factor in confidence['quality_factors']:
            lines.append(f"    - {factor}")

        # АНАЛИЗ КОМПОНЕНТ ОШИБКИ
        if error_analysis:
            lines.append(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ АНОМАЛИИ:")
            lines.append(f"  • Диагноз: {error_analysis['diagnosis']}")
            lines.append(f"  • Топ-3 параметра с наибольшим отклонением:")
            for i, comp in enumerate(error_analysis['top_3_contributors'], 1):
                lines.append(
                    f"    {i}. {comp['parameter']}: "
                    f"откл. {comp['absolute_error']:.6f}{comp['unit']} "
                    f"({comp['relative_error']:.1f}%)"
                )

    # УГЛЫ КОРРЕКЦИИ (только для спутников)
    if corrections:
        lines.append(f"\n🎯 РЕКОМЕНДУЕМЫЕ УГЛЫ И ВЕЛИЧИНЫ КОРРЕКЦИИ:")

        inc_corr = corrections['inclination']
        if abs(inc_corr['delta_degrees']) > 0.01:
            lines.append(f"  • Коррекция наклонения (North-South):")
            lines.append(f"    - Текущее: {inc_corr['current']:.4f}°")
            lines.append(f"    - Целевое: {inc_corr['target']:.4f}°")
            lines.append(f"    - Требуется изменение: {inc_corr['delta_degrees']:.4f}° ({inc_corr['direction']})")
            lines.append(f"    - Delta-V: {inc_corr['delta_v_ms']:.2f} м/с")

        ecc_corr = corrections['eccentricity']
        if abs(ecc_corr['delta']) > 0.0001:
            lines.append(f"  • Коррекция эксцентриситета:")
            lines.append(f"    - Текущее: {ecc_corr['current']:.6f}")
            lines.append(f"    - Целевое: {ecc_corr['target']:.6f}")
            lines.append(f"    - Требуется изменение: {ecc_corr['delta']:.6f}")
            lines.append(f"    - Delta-V: {ecc_corr['delta_v_ms']:.2f} м/с")

        drift_corr = corrections['drift']
        if abs(drift_corr['delta_degrees_per_day']) > 0.1:
            lines.append(f"  • Коррекция дрейфа (East-West):")
            lines.append(f"    - Текущий дрейф: {drift_corr['delta_degrees_per_day']:.3f}°/день")
            lines.append(f"    - Направление коррекции: {drift_corr['direction']}")
            lines.append(f"    - Delta-V: {drift_corr['delta_v_ms']:.2f} м/с")

        lines.append(f"  • Итого Delta-V: {corrections['total']['delta_v_ms']:.2f} м/с")
        lines.append(f"  • Оптимальное время коррекции: {corrections['total']['optimal_timing']}")
        lines.append(f"  • Точка выполнения маневра: {corrections['total']['burn_location']}")
    
    # Контекстные блоки в зависимости от типа объекта
    if is_debris:
        # Для мусора показываем анализ деградации
        degradation = assess_debris_degradation(tle_struct, tle_history)
        lines.append(f"\n📉 АНАЛИЗ ДЕГРАДАЦИИ ОРБИТЫ (Космический мусор):")
        lines.append(f"  • Срочность: {degradation['urgency']}")
        for rec in degradation['recommendations']:
            lines.append(f"  • {rec}")
    else:
        # Для спутников показываем station-keeping
        station_keeping = assess_station_keeping_for_satellite(inclination, eccentricity, mean_motion)
        fuel_budget = calculate_fuel_budget(inclination, eccentricity, mean_motion)
        
        lines.append(f"\n🛰️ КОРРЕКЦИЯ ОРБИТЫ (Station-Keeping):")
        lines.append(f"  • Срочность: {station_keeping['urgency']}")
        for rec in station_keeping['recommendations']:
            lines.append(f"  • {rec}")
        
        lines.append(f"\n⛽ ОЦЕНКА РАСХОДА ТОПЛИВА:")
        lines.append(f"  • Общий Delta-V: {fuel_budget['total_delta_v']:.2f} м/с")
        lines.append(f"  • Статус: {fuel_budget['budget_status']}")
        lines.append(f"  • Коррекция наклонения: {fuel_budget['breakdown']['inclination_correction']:.2f} м/с")
        lines.append(f"  • Коррекция эксцентриситета: {fuel_budget['breakdown']['eccentricity_correction']:.2f} м/с")
    
    # Риск столкновения (для всех)
    lines.append(f"\n⚠️ РИСК ОПАСНОГО СБЛИЖЕНИЯ:")
    lines.append(f"  • Уровень риска: {conjunction['risk_level']}")
    for warning in conjunction['warnings']:
        lines.append(f"  • {warning}")
    
    # Операционный статус
    lines.append(f"\n🔧 ОПЕРАЦИОННЫЙ СТАТУС:")
    lines.append(f"  • Статус: {operational['status']}")
    for warning in operational['warnings']:
        lines.append(f"  • {warning}")
    
    # Контекстные рекомендации
    lines.append(f"\n💡 ИТОГОВЫЕ РЕКОМЕНДАЦИИ (Контекстно-зависимые):")
    for rec in context_recommendations:
        lines.append(f"\n  {rec['priority']}: {rec['action']}")
        for detail in rec['details']:
            lines.append(f"    • {detail}")
    
    lines.append("\n" + "=" * 70)
    
    return "\n".join(lines)


def generate_summary_text(tle_struct):
    """Генерирует краткий summary"""
    name = tle_struct.get("name", "Неизвестный")
    mean_motion = tle_struct.get("mean_motion", 0)
    inclination = tle_struct.get("inclination", 0)
    eccentricity = tle_struct.get("eccentricity", 0)
    object_type = tle_struct.get("object_type", ObjectType.UNKNOWN)
    
    orbit_type = classify_orbit(inclination, mean_motion, eccentricity)
    period = calculate_orbital_period(mean_motion)
    
    type_marker = "🛰️" if object_type == ObjectType.SATELLITE else "🗑️"
    
    summary = (
        f"{type_marker} {name} | {orbit_type} | "
        f"Период: {period:.1f}мин | "
        f"Накл: {inclination:.2f}° | "
        f"Тип: {object_type.value}"
    )
    
    return summary


if __name__ == "__main__":
    # Тестовый пример
    test_tle_debris = {
        "name": "COSMOS 1408 DEB",
        "catalog_number": "49863",
        "intl_designator": "82092AAA",
        "object_type": ObjectType.DEBRIS,
        "status_by_name": OperationalStatus.FRAGMENTED,
        "inclination": 82.5,
        "raan": 45.0,
        "eccentricity": 0.0234,
        "argument_perigee": 90.0,
        "mean_anomaly": 180.0,
        "mean_motion": 14.85,
        "bstar": 0.00045,
        "mean_motion_dot": 0.00012,
        "epoch": {"year": 2024, "day_of_year": 1, "datetime": datetime.now()}
    }
    
    text = generate_detailed_text_with_values(test_tle_debris, reconstruction_error=0.025, threshold=0.01)
    print(text)
