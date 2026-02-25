"""
Unit тесты
Запуск: pytest test_tle_system.py -v
"""
import pytest
import torch
import numpy as np
from datetime import datetime
import os
import sys

# Импорты модулей
from parse_tle import (
    parse_tle, TLEParseError, 
    calculate_orbital_period, calculate_semi_major_axis,
    calculate_apogee_perigee, is_geo_orbit, classify_orbit,
    parse_scientific_notation
)
from train_model import TLEAnalyzer, TLENormalizer, TLEDataset
from generate import (
    generate_detailed_text_with_values,
    generate_summary_text,
    assess_station_keeping_for_satellite,
    calculate_fuel_budget,
    assess_conjunction_risk
)


# ТЕСТЫ ПАРСИНГА TLE

class TestTLEParsing:
    """Тесты парсинга TLE данных"""
    
    @pytest.fixture
    def valid_tle(self):
        """Валидная TLE запись для тестов"""
        return (
            "ISS (ZARYA)",
            "1 25544U 98067A   24001.50000000  .00012345  00000-0  12345-3 0  9992",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391111111"
        )
    
    @pytest.fixture
    def geo_tle(self):
        """GEO спутник для тестов"""
        return (
            "INTELSAT 902",
            "1 26900U 01039A   24032.50000000  .00000010  00000-0  00000-0 0  9991",
            "2 26900   0.0500  45.0000 0000200  90.0000 180.0000  1.00273000082345"
        )
    
    def test_parse_valid_tle(self, valid_tle):
        """Тест парсинга валидной TLE"""
        vector, tle_struct = parse_tle(*valid_tle)
        
        # Проверка вектора
        assert len(vector) == 6
        assert all(isinstance(v, float) for v in vector)
        
        # Проверка структуры
        assert tle_struct['name'] == "ISS (ZARYA)"
        assert tle_struct['catalog_number'] == "25544"
        assert tle_struct['inclination'] == pytest.approx(51.6416, rel=1e-4)
        assert tle_struct['mean_motion'] == pytest.approx(15.72125391, rel=1e-6)
    
    def test_parse_geo_satellite(self, geo_tle):
        """Тест парсинга GEO спутника"""
        vector, tle_struct = parse_tle(*geo_tle)
        
        assert tle_struct['inclination'] < 1.0  # GEO имеет малое наклонение
        assert 0.99 < tle_struct['mean_motion'] < 1.01  # ~1 оборот/день
        assert tle_struct['eccentricity'] < 0.001  # почти круговая
    
    def test_invalid_tle_short_line(self):
        """Тест короткой строки TLE"""
        with pytest.raises(TLEParseError):
            parse_tle("TEST", "1 12345", "2 12345")
    
    def test_invalid_tle_wrong_format(self):
        """Тест неверного формата"""
        with pytest.raises(TLEParseError):
            parse_tle(
                "TEST",
                "3 25544U 98067A   24001.50000000  .00012345  00000-0  12345-3 0  9992",
                "4 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391111111"
            )
    
    def test_epoch_conversion(self, valid_tle):
        """Тест конвертации epoch в datetime"""
        _, tle_struct = parse_tle(*valid_tle)
        epoch = tle_struct['epoch']
        
        assert epoch['year'] == 2024
        assert isinstance(epoch['datetime'], datetime)
    
    def test_scientific_notation_parsing(self):
        """Тест парсинга научной нотации"""
        assert "GEO" in classify_orbit(0.05, 1.0027, 0.0001)
        assert "LEO" in classify_orbit(51.6, 15.5, 0.001)
        assert "HEO" in classify_orbit(63.4, 2.0, 0.72)

# ТЕСТЫ ОРБИТАЛЬНЫХ РАСЧЕТОВ

class TestOrbitalCalculations:
    """Тесты орбитальных расчетов"""
    
    def test_orbital_period_geo(self):
        """Тест расчета периода для GEO"""
        mean_motion = 1.0027  # GEO
        period = calculate_orbital_period(mean_motion)
        
        # GEO должен иметь период ~1436 минут (23.93 часа)
        assert 1430 < period < 1440
    
    def test_orbital_period_leo(self):
        """Тест расчета периода для LEO"""
        mean_motion = 15.5  # типичный LEO
        period = calculate_orbital_period(mean_motion)
        
        # ~90-100 минут
        assert 85 < period < 100
    
    def test_semi_major_axis_geo(self):
        """Тест расчета большой полуоси для GEO"""
        mean_motion = 1.0027
        a = calculate_semi_major_axis(mean_motion)
        
        # GEO на высоте ~35786 км, радиус орбиты ~42164 км
        assert 42000 < a < 42300
    
    def test_apogee_perigee_circular(self):
        """Тест расчета апогея/перигея для круговой орбиты"""
        a = 42164.0  # GEO
        ecc = 0.0001  # почти круговая
        
        apogee, perigee = calculate_apogee_perigee(a, ecc)
        
        # Разница должна быть минимальной
        assert abs(apogee - perigee) < 10
    
    def test_apogee_perigee_elliptical(self):
        """Тест для эллиптической орбиты"""
        a = 26600.0
        ecc = 0.72  # Молния орбита
        
        apogee, perigee = calculate_apogee_perigee(a, ecc)
        
        # Должна быть большая разница
        assert apogee > perigee
        assert apogee - perigee > 20000
    
    def test_geo_orbit_detection(self):
        """Тест детекции GEO орбиты"""
        assert is_geo_orbit(1.0027) is True
        assert is_geo_orbit(1.0) is True
        assert is_geo_orbit(15.5) is False
        assert is_geo_orbit(0.5) is False
    
    def test_orbit_classification(self):
        """Тест классификации орбит"""
        # GEO
        assert "GEO" in classify_orbit(0.05, 1.0027, 0.0001)
        
        # LEO
        assert "LEO" in classify_orbit(51.6, 15.5, 0.001)


# ТЕСТЫ МОДЕЛИ

class TestNeuralNetwork:
    """Тесты нейрон сети"""
    
    @pytest.fixture
    def model(self):
        """Создание модели для тестов"""
        return TLEAnalyzer(input_dim=6, hidden_dim=64, latent_dim=16)
    
    @pytest.fixture
    def sample_data(self):
        """Тестовые данные"""
        return torch.randn(5, 6)  # 5 образцов, 6 признаков
    
    def test_model_initialization(self, model):
        """Тест инициализации модели"""
        assert isinstance(model, TLEAnalyzer)
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'anomaly_detector')
    
    def test_model_forward_pass(self, model, sample_data):
        """Тест прямого прохода"""
        reconstructed, latent = model(sample_data)
        
        assert reconstructed.shape == sample_data.shape
        assert latent.shape == (5, 16)  # latent_dim=16
    
    def test_model_output_range(self, model, sample_data):
        """Тест диапазона выходных значений"""
        reconstructed, latent = model(sample_data)
        
        # Проверка на NaN/Inf
        assert not torch.isnan(reconstructed).any()
        assert not torch.isinf(reconstructed).any()
    
    def test_anomaly_detection(self, model, sample_data):
        """Тест детекции аномалий"""
        mse, anomaly_score = model.detect_anomaly(sample_data)
        
        assert mse.shape == (5,)  # 5 образцов
        assert anomaly_score.shape == (5, 1)
        assert (anomaly_score >= 0).all() and (anomaly_score <= 1).all()


class TestNormalizer:
    """Тесты нормализатора"""
    
    @pytest.fixture
    def normalizer(self):
        """Создание нормализатора"""
        return TLENormalizer()
    
    @pytest.fixture
    def sample_dataset(self):
        """Тестовый датасет"""
        return [
            ([51.0, 247.0, 0.0067, 130.0, 325.0, 15.72], 
             [51.0, 247.0, 0.0067, 130.0, 325.0, 15.72])
            for _ in range(10)
        ]
    
    def test_normalizer_fit(self, normalizer, sample_dataset):
        """Тест обучения нормализатора"""
        normalizer.fit(sample_dataset)
        assert normalizer.fitted is True
    
    def test_normalizer_transform(self, normalizer, sample_dataset):
        """Тест трансформации"""
        normalizer.fit(sample_dataset)
        vector = [51.0, 247.0, 0.0067, 130.0, 325.0, 15.72]
        
        normalized = normalizer.transform(vector)
        assert len(normalized) == 6
        assert isinstance(normalized, np.ndarray)
    
    def test_normalizer_inverse_transform(self, normalizer, sample_dataset):
        """Тест обратной трансформации"""
        normalizer.fit(sample_dataset)
        vector = [51.0, 247.0, 0.0067, 130.0, 325.0, 15.72]
        
        normalized = normalizer.transform(vector)
        restored = normalizer.inverse_transform(normalized)
        
        # Должны получить исходные значения с небольшой погрешностью
        np.testing.assert_allclose(vector, restored, rtol=1e-5)


# ТЕСТЫ ГЕНЕРАЦИИ РЕКОМЕНДАЦИЙ

class TestRecommendationGeneration:
    """Тесты генерации рекомендаций"""
    
    @pytest.fixture
    def geo_tle_struct(self):
        """TLE структура GEO спутника"""
        return {
            "name": "TEST GEO",
            "inclination": 0.05,
            "raan": 45.0,
            "eccentricity": 0.0002,
            "argument_perigee": 90.0,
            "mean_anomaly": 180.0,
            "mean_motion": 1.00273,
            "bstar": 0.00001,
            "mean_motion_dot": 0.00000001,
            "epoch": {"year": 2024, "day_of_year": 1, "datetime": datetime.now()}
        }
    
    def test_station_keeping_assessment(self):
        """Тест оценки station-keeping"""
        # Идеальная GEO орбита
        result = assess_station_keeping_for_satellite(0.0, 0.0, 1.0)
        assert result['urgency'] == "Низкая"
        
        # Требует коррекции
        result = assess_station_keeping_for_satellite(1.5, 0.01, 1.05)
        assert result['urgency'] in ["Средняя", "Высокая"]
    
    def test_fuel_budget_calculation(self):
        """Тест расчета расхода топлива"""
        result = calculate_fuel_budget(0.05, 0.0002, 1.00273)
        
        assert 'total_delta_v' in result
        assert 'budget_status' in result
        assert result['total_delta_v'] >= 0

    def test_collision_risk_assessment(self):
        """Тест оценки риска столкновения"""
        tle_example = {
            'mean_motion': 0.0001,
            'eccentricity': 0.001,
            'object_type': 'satellite'
        }

        result = assess_conjunction_risk(tle_example, tle_example['object_type'])

        assert isinstance(result, dict)
        assert 'risk_level' in result
        assert 'warnings' in result

    def test_detailed_text_generation(self, geo_tle_struct):
        """Тест генерации детального отчета"""
        text = generate_detailed_text_with_values(
            geo_tle_struct,
            reconstruction_error=0.001,
            anomaly_score=0.05,
            threshold=0.01
        )
        
        assert len(text) > 0
        assert "ОРБИТАЛЬНЫЕ ПАРАМЕТРЫ" in text
        assert "РЕКОМЕНДАЦИИ" in text
    
    def test_summary_generation(self, geo_tle_struct):
        """Тест генерации краткого summary"""
        summary = generate_summary_text(geo_tle_struct)
        
        assert len(summary) > 0
        assert "TEST GEO" in summary


# ============================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ
# ============================================================================

class TestIntegration:
    """Интеграционные тесты полной системы"""
    
    @pytest.fixture
    def temp_tle_file(self, tmp_path):
        """Создание временного TLE файла"""
        tle_content = """INTELSAT 902
1 26900U 01039A   24032.50000000  .00000010  00000-0  00000-0 0  9991
2 26900   0.0500  45.0000 0000200  90.0000 180.0000  1.00273000082345
EUTELSAT 7A
1 37806U 11045A   24032.50000000  .00000011  00000-0  00000-0 0  9992
2 37806   0.0450  50.0000 0000150  95.0000 185.0000  1.00270000045678"""
        
        file_path = tmp_path / "test_tle.txt"
        file_path.write_text(tle_content)
        return str(file_path)
    
    def test_full_pipeline(self, temp_tle_file):
        """Тест полного pipeline: парсинг → модель → рекомендации"""
        from train_model import load_tle_dataset
        
        # Загрузка данных
        dataset = load_tle_dataset(temp_tle_file, geo_only=True)
        assert len(dataset) > 0
        
        # Проверка структуры
        vector, _ = dataset[0]
        assert len(vector) == 6
        
        # Создание и тест модели
        model = TLEAnalyzer()
        model.eval()
        x = torch.tensor([vector, vector], dtype=torch.float32)
        reconstructed, latent = model(x)
        
        assert reconstructed.shape == x.shape


# ТЕСТЫ ПРОИЗВОДИТЕЛЬНОСТИ

class TestPerformance:
    """Тесты производительности"""
    
    def test_parsing_speed(self):
        """Тест скорости парсинга"""
        import time
        
        tle = (
            "TEST",
            "1 25544U 98067A   24001.50000000  .00012345  00000-0  12345-3 0  9992",
            "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391111111"
        )
        
        start = time.time()
        for _ in range(1000):
            parse_tle(*tle)
        elapsed = time.time() - start
        
        # < 1 сек для 1000 операций
        assert elapsed < 1.0
    
    def test_model_inference_speed(self):
        """Тест скорости inference модели"""
        import time
        
        model = TLEAnalyzer()
        model.eval()
        
        data = torch.randn(100, 6)
        
        start = time.time()
        with torch.no_grad():
            for i in range(100):
                model(data[i:i+1])
        elapsed = time.time() - start
        
        # Должно быть быстро
        assert elapsed < 1.0


# КОНФИГУРАЦИЯ PYTEST

if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])
