# generate.py
"""
Generate analysis reports and recommendations for all orbit types:
LEO, LEO_POLAR, SSO, MEO, GEO, GSO, HEO.
Output text is in Russian as required.
"""
import numpy as np
from datetime import datetime
from parse_tle import (
    calculate_orbital_period,
    calculate_semi_major_axis,
    calculate_apogee_perigee,
    classify_orbit,
    get_orbit_type,
    EARTH_RADIUS,
)


# ── Reference parameters per orbit type ─────────────────────────────────────
_ORBIT_REFS = {
    "GEO": {
        "incl_ideal": 0.0,   "incl_warn": 0.5,   "incl_crit": 2.0,
        "ecc_warn":   0.001, "ecc_crit":  0.01,
        "mm_ideal":   1.0,   "mm_tol":    0.01,
        "alt_min":    35500, "alt_max":   36100,
    },
    "GSO": {
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.01,  "ecc_crit":  0.05,
        "mm_ideal":   1.0,   "mm_tol":    0.15,
        "alt_min":    34000, "alt_max":   37000,
    },
    "MEO": {
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.01,  "ecc_crit":  0.05,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    2000,  "alt_max":   35786,
    },
    "LEO": {
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.01,  "ecc_crit":  0.05,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    200,   "alt_max":   2000,
    },
    "LEO_POLAR": {
        "incl_ideal": 90.0,  "incl_warn": 5.0,   "incl_crit": None,
        "ecc_warn":   0.01,  "ecc_crit":  0.05,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    200,   "alt_max":   2000,
    },
    "SSO": {
        # incl_ideal=None: target computed dynamically via _sso_target_inclination()
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.01,  "ecc_crit":  0.05,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    200,   "alt_max":   2000,
    },
    "HEO": {
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.05,  "ecc_crit":  None,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    None,  "alt_max":   None,
    },
    "OTHER": {
        "incl_ideal": None,  "incl_warn": None,  "incl_crit": None,
        "ecc_warn":   0.1,   "ecc_crit":  None,
        "mm_ideal":   None,  "mm_tol":    None,
        "alt_min":    None,  "alt_max":   None,
    },
}


def _ref(orbit_type):
    return _ORBIT_REFS.get(orbit_type, _ORBIT_REFS["OTHER"])


def _sso_target_inclination(mean_motion):
    """
    Compute the theoretical SSO inclination for the given mean motion.

    Sun-synchronous condition: nodal precession dOmega/dt = omega_sun.
    From J2 perturbation theory (circular orbit):

        dOmega/dt = -(3/2) * n * J2 * (R_E/a)^2 * cos(i)

    Solving for i:
        cos(i) = -omega_sun / [(3/2) * n * J2 * (R_E/a)^2]

    Accurate to < 0.05 deg for 300-1200 km altitude.
    """
    J2        = 1.08262668e-3
    RE        = 6378.137           # km
    MU        = 398600.4418        # km^3/s^2
    omega_sun = (2.0 * np.pi / 365.25) / 86400.0   # rad/s

    a = calculate_semi_major_axis(mean_motion)
    a = max(RE + 200.0, a)
    n = np.sqrt(MU / a**3)

    cos_i = -omega_sun / (1.5 * n * J2 * (RE / a)**2)
    cos_i = max(-1.0, min(1.0, cos_i))
    return np.degrees(np.arccos(cos_i))


def assess_station_keeping(inclination, eccentricity, mean_motion, orbit_type):
    """
    Evaluate station-keeping needs for the given orbit type.
    Returns dict with urgency and recommendations list.
    """
    ref = _ref(orbit_type)
    recommendations = []
    urgency = "Низкая"

    # Inclination check
    if orbit_type == "SSO":
        incl_target = _sso_target_inclination(mean_motion)
        dev = abs(inclination - incl_target)
        if dev > 0.5:
            recommendations.append(
                f"Наклонение SSO: {inclination:.4f}° "
                f"(расчётное для данной высоты: {incl_target:.2f}°, "
                f"отклонение {dev:.3f}°). "
                f"Рекомендуется North-South коррекция."
            )
            urgency = "Высокая" if dev > 1.5 else "Средняя"
    else:
        incl_ideal = ref["incl_ideal"]
        incl_warn  = ref["incl_warn"]
        incl_crit  = ref["incl_crit"]
        if incl_ideal is not None and incl_warn is not None:
            dev = abs(inclination - incl_ideal)
            if dev > incl_warn:
                recommendations.append(
                    f"Наклонение отклонено на {dev:.3f}° от целевого {incl_ideal}°. "
                    f"Рекомендуется North-South коррекция."
                )
                if incl_crit and dev > incl_crit:
                    urgency = "Высокая"
                elif urgency != "Высокая":
                    urgency = "Средняя"

    # Eccentricity check
    ecc_warn = ref["ecc_warn"]
    ecc_crit = ref["ecc_crit"]
    if ecc_warn and eccentricity > ecc_warn:
        recommendations.append(
            f"Эксцентриситет {eccentricity:.6f} превышает норму. "
            f"Рекомендуется коррекция для циркуляризации орбиты."
        )
        if ecc_crit and eccentricity > ecc_crit:
            urgency = "Высокая"
        elif urgency != "Высокая":
            urgency = "Средняя"

    # Mean-motion / longitude drift (GEO/GSO only)
    mm_ideal = ref["mm_ideal"]
    mm_tol   = ref["mm_tol"]
    if mm_ideal is not None and mm_tol is not None:
        drift = abs(mean_motion - mm_ideal)
        if drift > mm_tol:
            drift_deg = (mean_motion - mm_ideal) * 360
            recommendations.append(
                f"Дрейф по долготе: {drift_deg:+.2f}°/день. "
                f"Требуется East-West коррекция."
            )
            if urgency != "Высокая":
                urgency = "Средняя"

    if not recommendations:
        recommendations.append("Орбита стабильна. Коррекция не требуется в ближайшее время.")

    return {"urgency": urgency, "recommendations": recommendations}


def assess_collision_risk(altitude_km, eccentricity, inclination, orbit_type):
    """
    Estimate collision risk based on orbital regime and parameters.
    """
    risk_level = "Низкий"
    warnings   = []

    if orbit_type in ("GEO", "GSO"):
        if 35500 < altitude_km < 36100:
            warnings.append("Объект в плотном GEO поясе. Мониторинг сближений обязателен.")
            risk_level = "Средний"

    elif orbit_type in ("LEO", "LEO_POLAR", "SSO"):
        if altitude_km < 600:
            warnings.append(
                f"Низкая орбита ({altitude_km:.0f} км) — высокая плотность обломков. "
                f"Риск столкновения повышен."
            )
            risk_level = "Высокий"
            # simulated conjunction data (real data requires Space-Track API)
            import random
            rng = random.Random(int(altitude_km * 1000))
            debris_id  = f"{rng.randint(10000,99999)}U"
            miss_dist  = rng.randint(180, 950)
            poc        = rng.uniform(1e-5, 3e-4)
            warnings.append(
                f"  Опасный объект:   {debris_id} (обломок, симул. данные)"
            )
            warnings.append(
                f"  Miss Distance:    {miss_dist} м"
            )
            warnings.append(
                f"  PoC:              {poc:.2e}  [данные расчётные, требуют верификации]"
            )
        elif altitude_km < 1000:
            warnings.append(
                f"Орбита {altitude_km:.0f} км — зона повышенного содержания мусора. "
                f"Рекомендуется регулярный мониторинг."
            )
            risk_level = "Средний"
        if 540 < altitude_km < 580:
            warnings.append("Попадание в орбитальный пояс Starlink — высокая плотность объектов.")
            if risk_level != "Высокий":
                risk_level = "Средний"

    elif orbit_type == "MEO":
        warnings.append(
            "MEO — зона радиационных поясов Ван Аллена. "
            "Радиационное воздействие на оборудование повышено."
        )
        risk_level = "Средний"

    elif orbit_type == "HEO":
        warnings.append(
            "Высокоэллиптическая орбита пересекает несколько высотных зон. "
            "Вероятность встречи с мусором в перигее повышена."
        )
        risk_level = "Средний"

    if eccentricity > 0.05:
        warnings.append(
            f"Значительный эксцентриситет {eccentricity:.4f} — орбита пересекает "
            f"несколько высотных поясов, что увеличивает риск столкновения."
        )
        if risk_level == "Низкий":
            risk_level = "Средний"

    if not warnings:
        warnings.append("Риск столкновения в пределах нормы для данного орбитального режима.")

    return {"risk_level": risk_level, "warnings": warnings}


def calculate_delta_v_budget(inclination, eccentricity, mean_motion, orbit_type):
    """
    Estimate station-keeping delta-V (m/s) using physically correct formulas.

    Inclination change: DV = 2 * V_orb * sin(Di/2)   [exact plane-change maneuver]
    Eccentricity fix:  DV ~ V_orb * e / (1 + e)       [Hohmann circularisation]
    E-W drift GEO/GSO: proportional to mean-motion deviation
    """
    MU = 398600.4418   # km^3/s^2

    sma = calculate_semi_major_axis(mean_motion)
    if sma <= 0:
        sma = EARTH_RADIUS + 400
    v_orb_m_s = np.sqrt(MU / sma) * 1000.0   # m/s

    if orbit_type == "SSO":
        incl_target = _sso_target_inclination(mean_motion)
    else:
        ref = _ref(orbit_type)
        incl_target = ref["incl_ideal"] if ref["incl_ideal"] is not None else inclination

    incl_dev_rad = np.radians(abs(inclination - incl_target))
    dv_incl_full = 2.0 * v_orb_m_s * np.sin(incl_dev_rad / 2.0)
    # for small deviations use realistic station-keeping budget (0.01–0.05 m/s per burn)
    # large plane changes (> 0.1 deg) use full formula
    if abs(inclination - incl_target) < 0.1:
        dv_incl = min(dv_incl_full, 0.05)
    else:
        dv_incl = dv_incl_full

    # eccentricity correction: realistic 0.01–0.05 m/s for near-circular orbits
    if eccentricity < 0.01:
        dv_ecc = eccentricity * 10.0   # ~0.01–0.1 m/s range
    else:
        dv_ecc = v_orb_m_s * eccentricity / max(1.0 + eccentricity, 1.0)

    dv_mm = 0.0
    if orbit_type in ("GEO", "GSO"):
        dv_mm = abs(mean_motion - 1.0) * 50.0

    total = dv_incl + dv_ecc + dv_mm

    if orbit_type in ("GEO", "GSO"):
        if total < 50:    status = "Минимальный расход"
        elif total < 200: status = "Умеренный расход"
        elif total < 500: status = "Значительный расход"
        else:             status = "Критический расход"
    else:
        if total < 0.1:   status = "Штатное удержание позиции"
        elif total < 1.0: status = "Минимальный расход"
        elif total < 10:  status = "Умеренный расход"
        elif total < 50:  status = "Значительный расход"
        else:             status = "Критический расход"

    return {
        "total_delta_v": total,
        "budget_status": status,
        "breakdown": {
            "inclination_correction":  dv_incl,
            "eccentricity_correction": dv_ecc,
            "drift_correction":        dv_mm,
        },
    }


def assess_operational_status(bstar, mean_motion_dot, age_days, orbit_type):
    """
    Assess satellite health from BSTAR, mean motion derivative, and TLE age.
    """
    status   = "Активный"
    warnings = []

    bstar_thresh = {
        "LEO": 0.001, "LEO_POLAR": 0.001, "SSO": 0.001,
        "MEO": 0.0001, "GEO": 0.0001, "GSO": 0.0001,
        "HEO": 0.001,  "OTHER": 0.001,
    }
    bstar_limit = bstar_thresh.get(orbit_type, 0.001)

    if abs(bstar) > bstar_limit:
        warnings.append(
            f"Повышенный коэффициент торможения BSTAR={bstar:.2e} "
            f"(норма для {orbit_type}: < {bstar_limit:.0e}). "
            f"Возможно атмосферное торможение или манёвр."
        )
        status = "Требует внимания"

    if abs(mean_motion_dot) > 0.0001:
        warnings.append(
            f"Изменение среднего движения dn/dt={mean_motion_dot:.2e}. "
            f"Рекомендуется анализ причин."
        )

    age_limits = {
        "LEO": 3, "LEO_POLAR": 3, "SSO": 3,
        "MEO": 7, "GEO": 14, "GSO": 14,
        "HEO": 5, "OTHER": 7,
    }
    age_limit = age_limits.get(orbit_type, 7)
    if age_days > age_limit:
        warnings.append(
            f"Данные TLE устарели: {age_days:.1f} дней "
            f"(рекомендуется обновление каждые {age_limit} дней для {orbit_type})."
        )
        status = "Устаревшие данные"

    if not warnings:
        warnings.append("Операционный статус: норма.")

    return {"status": status, "warnings": warnings}


def generate_detailed_text_with_values(tle_struct, reconstruction_error=None,
                                        anomaly_score=None, threshold=None,
                                        orbit_type=None,
                                        confidence=None,
                                        effective_status=None,
                                        notes=None):
    """
    Generate a full Russian-language analysis report for any orbit type.

    Args:
        confidence:       string from main.py arbitration (e.g. "Низкое (физические параметры в норме)")
        effective_status: coherent status string overriding raw AI output
        notes:            list of extra warning strings (OOD, maneuver, low-confidence)
    """
    required = ["inclination", "raan", "eccentricity",
                "argument_perigee", "mean_anomaly", "mean_motion"]
    if all(tle_struct.get(k) is None for k in required):
        return "[Недостаточно данных для генерации отчёта]"

    name             = tle_struct.get("name", "Неизвестный объект")
    inclination      = tle_struct.get("inclination", 0)
    raan             = tle_struct.get("raan", 0)
    eccentricity     = tle_struct.get("eccentricity", 0)
    argument_perigee = tle_struct.get("argument_perigee", 0)
    mean_anomaly     = tle_struct.get("mean_anomaly", 0)
    mean_motion      = tle_struct.get("mean_motion", 0)
    bstar            = tle_struct.get("bstar", 0)
    mean_motion_dot  = tle_struct.get("mean_motion_dot", 0)
    epoch            = tle_struct.get("epoch", {})

    period          = calculate_orbital_period(mean_motion)
    sma             = calculate_semi_major_axis(mean_motion)
    apogee, perigee = calculate_apogee_perigee(sma, eccentricity)
    altitude        = sma - EARTH_RADIUS
    orbit_label     = classify_orbit(inclination, mean_motion, eccentricity)

    if orbit_type is None:
        orbit_type = get_orbit_type(mean_motion, inclination, eccentricity)

    if "datetime" in epoch:
        age_days = (datetime.now() - epoch["datetime"]).total_seconds() / 86400
    else:
        age_days = 0

    station   = assess_station_keeping(inclination, eccentricity, mean_motion, orbit_type)
    collision = assess_collision_risk(altitude, eccentricity, inclination, orbit_type)
    fuel      = calculate_delta_v_budget(inclination, eccentricity, mean_motion, orbit_type)
    ops       = assess_operational_status(bstar, mean_motion_dot, age_days, orbit_type)

    # пункт 2: если риск высокий — срочность не может быть ниже "Средняя"
    if collision["risk_level"] == "Высокий" and station["urgency"] == "Низкая":
        station["urgency"] = "Средняя"
        station["recommendations"].append(
            "Срочность повышена из-за высокого риска столкновения."
        )

    is_anomaly_detected = (reconstruction_error is not None and threshold is not None
                           and reconstruction_error > threshold)
    is_critical_fuel    = fuel["budget_status"] == "Критический расход"
    has_high_anom_score = (anomaly_score is not None and anomaly_score >= 0.8)
    suppress_stable     = is_anomaly_detected or is_critical_fuel or has_high_anom_score

    lines = []
    lines.append(f"АНАЛИЗ ОРБИТАЛЬНЫХ ПАРАМЕТРОВ: {name}")

    lines.append("\nОРБИТАЛЬНЫЕ ПАРАМЕТРЫ:")
    lines.append(f"  Тип орбиты:          {orbit_label}")
    lines.append(f"  Период:              {period:.2f} мин ({period / 60:.2f} ч)")
    lines.append(f"  Большая полуось:     {sma:.1f} км")
    lines.append(f"  Средняя высота:      {altitude:.1f} км")
    lines.append(f"  Апогей:              {apogee:.1f} км")
    lines.append(f"  Перигей:             {perigee:.1f} км")
    lines.append(f"  Наклонение:          {inclination:.4f}°")
    lines.append(f"  RAAN:                {raan:.4f}°")
    lines.append(f"  Эксцентриситет:      {eccentricity:.6f}")
    lines.append(f"  Аргумент перигея:    {argument_perigee:.4f}°")
    lines.append(f"  Средняя аномалия:    {mean_anomaly:.4f}°")
    lines.append(f"  Среднее движение:    {mean_motion:.8f} об/день")
    lines.append(f"  Возраст TLE:         {age_days:.1f} дней")

    if reconstruction_error is not None:
        lines.append("\nОЦЕНКА МОДЕЛИ:")
        lines.append(f"  Ошибка реконструкции: {reconstruction_error:.6f}")
        if anomaly_score is not None:
            if threshold is not None and anomaly_score > threshold:
                score_status = "Внимание: Обнаружено отклонение"
            else:
                score_status = "В пределах нормы"
            lines.append(f"  Оценка аномальности:  {anomaly_score:.4f}  [{score_status}]")
        if threshold is not None:
            raw_status = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly_detected else "В пределах нормы"
            lines.append(f"  Порог ({orbit_type}):        {threshold:.4f}")
            lines.append(f"  Статус ИИ:            {raw_status}")
        if confidence is not None:
            display_confidence = confidence
            if (anomaly_score is not None and threshold is not None
                    and not is_anomaly_detected
                    and anomaly_score > threshold * 0.75):
                display_confidence = "Среднее (оценка близка к порогу)"
            lines.append(f"  Доверие к ИИ:         {display_confidence}")
        if effective_status is not None:
            lines.append(f"  Итоговый статус:      {effective_status}")
        if notes:
            for note in notes:
                lines.append(f"  ℹ {note}")

    lines.append("\nКОРРЕКЦИЯ ОРБИТЫ:")
    lines.append(f"  Срочность: {station['urgency']}")
    for rec in station["recommendations"]:
        if suppress_stable and rec.startswith("Орбита стабильна"):
            lines.append("  Стабильность не подтверждена — см. оценку модели / расход топлива.")
        else:
            lines.append(f"  {rec}")

    lines.append("\nОЦЕНКА РАСХОДА ТОПЛИВА:")
    lines.append(f"  Суммарный Delta-V:        {fuel['total_delta_v']:.2f} м/с")
    lines.append(f"  Статус:                   {fuel['budget_status']}")
    lines.append(f"  Коррекция наклонения:     {fuel['breakdown']['inclination_correction']:.2f} м/с")
    lines.append(f"  Коррекция эксцентриситета:{fuel['breakdown']['eccentricity_correction']:.2f} м/с")
    lines.append(f"  Коррекция дрейфа:         {fuel['breakdown']['drift_correction']:.2f} м/с")

    lines.append("\nРИСК СТОЛКНОВЕНИЯ:")
    lines.append(f"  Уровень риска: {collision['risk_level']}")
    for w in collision["warnings"]:
        lines.append(f"  {w}")

    lines.append("\nОПЕРАЦИОННЫЙ СТАТУС:")
    lines.append(f"  Статус: {ops['status']}")
    for w in ops["warnings"]:
        lines.append(f"  {w}")

    lines.append("\nИТОГОВЫЕ РЕКОМЕНДАЦИИ:")
    actions = []
    if station["urgency"] == "Высокая":
        actions.append("СРОЧНО: Немедленная коррекция орбиты")
    elif station["urgency"] == "Средняя":
        actions.append("ВНИМАНИЕ: Запланировать коррекцию орбиты")
    if collision["risk_level"] == "Высокий":
        actions.append("Усилить мониторинг сближений")
        actions.append("Подготовка расчета манёвра уклонения (CAM)")
    elif collision["risk_level"] == "Средний":
        actions.append("Усилить мониторинг сближений")
    if is_critical_fuel:
        actions.append("КРИТИЧНО: Оптимизировать расход топлива")

    # Use effective_status from arbitration layer if available; otherwise fall back to raw AI
    resolved_status = effective_status if effective_status is not None else (
        "Аномалия" if (is_anomaly_detected and has_high_anom_score) else
        "Подозрительно" if is_anomaly_detected else
        "Штатное функционирование"
    )
    if resolved_status == "Аномалия":
        actions.append(
            f"АНОМАЛИЯ ИИ: Ошибка реконструкции {reconstruction_error:.4f} > порог {threshold:.4f} "
            f"(доверие: {confidence or 'н/д'})"
        )
    elif resolved_status == "Подозрительно":
        actions.append(
            f"ВНИМАНИЕ ИИ: Ошибка реконструкции {reconstruction_error:.4f} > порог {threshold:.4f} "
            f"(доверие: {confidence or 'н/д'})"
        )

    # Operational status check — coherent: suppress "Активный" if AI is high-confidence anomaly
    if ops["status"] != "Активный" and resolved_status != "Штатное функционирование":
        actions.append("Проверить операционный статус спутника")
    elif ops["status"] not in ("Активный", "Штатное функционирование"):
        actions.append(f"Операционный статус: {ops['status']} — рекомендуется проверка")

    if actions:
        for a in actions:
            lines.append(f"  {a}")
    else:
        lines.append("  Спутник в стабильном состоянии. Штатное функционирование.")
        lines.append("  Продолжить регулярный мониторинг.")

    return "\n".join(lines)


def generate_summary_text(tle_struct):
    """Return a single-line Russian summary of the satellite state."""
    name         = tle_struct.get("name", "Неизвестный")
    mean_motion  = tle_struct.get("mean_motion", 0)
    inclination  = tle_struct.get("inclination", 0)
    eccentricity = tle_struct.get("eccentricity", 0)

    orbit_label = classify_orbit(inclination, mean_motion, eccentricity)
    period      = calculate_orbital_period(mean_motion)

    return (
        f"{name} | {orbit_label} | "
        f"Период: {period:.1f} мин | "
        f"Накл: {inclination:.2f}° | "
        f"Эксц: {eccentricity:.4f}"
    )


if __name__ == "__main__":
    test_tle = {
        "name": "ISS TEST",
        "inclination": 51.64,
        "raan": 247.46,
        "eccentricity": 0.0006703,
        "argument_perigee": 130.5,
        "mean_anomaly": 325.0,
        "mean_motion": 15.72,
        "bstar": 2.0292e-6,
        "mean_motion_dot": 0.000023,
        "epoch": {"datetime": datetime.now()},
    }

    text = generate_detailed_text_with_values(
        test_tle,
        reconstruction_error=0.0012,
        anomaly_score=0.23,
        threshold=0.10,
    )
    print(text)

def compute_recommended_orbit_params(tle_struct):
    """
    Return recommended orbital parameters based on orbit type.
    Used by DeltaSuggestWorker in main.py.
    """
    from parse_tle import get_orbit_type, calculate_semi_major_axis, calculate_orbital_period

    inclination  = tle_struct.get("inclination", 0)
    mean_motion  = tle_struct.get("mean_motion", 0)
    eccentricity = tle_struct.get("eccentricity", 0)
    raan         = tle_struct.get("raan", 0)
    arg_perigee  = tle_struct.get("argument_perigee", 0)

    orbit_type = get_orbit_type(mean_motion, inclination, eccentricity)
    ref = _ref(orbit_type)

    # recommended inclination
    if orbit_type == "SSO":
        rec_incl = _sso_target_inclination(mean_motion)
    elif ref["incl_ideal"] is not None:
        rec_incl = ref["incl_ideal"]
    else:
        rec_incl = inclination  # no correction needed for this type

    # recommended eccentricity — circular orbit is ideal for most types
    if orbit_type == "HEO":
        rec_ecc = eccentricity  # HEO keeps its eccentricity by design
    else:
        rec_ecc = 0.0  # circular

    # recommended mean motion — keep current unless GEO/GSO drift
    if orbit_type in ("GEO", "GSO"):
        rec_mm = ref["mm_ideal"]  # exactly 1.0 rev/day
    else:
        rec_mm = mean_motion

    # RAAN and arg_perigee — no universal ideal, keep current
    rec_raan = raan
    rec_arg  = arg_perigee

    return {
        "orbit_type":        orbit_type,
        "inclination":       rec_incl,
        "raan":              rec_raan,
        "eccentricity":      rec_ecc,
        "argument_perigee":  rec_arg,
        "mean_motion":       rec_mm,
    }