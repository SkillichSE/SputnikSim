# main.py
"""
Главный скрипт для анализа TLE данных с использованием обученной модели
"""
import torch
import numpy as np
from train_model import TLEAnalyzer, TLENormalizer
from parse_tle import parse_tle, TLEParseError, is_geo_orbit
from generate import generate_detailed_text_with_values, generate_summary_text
import os
import argparse
from datetime import datetime


class TLEAnalysisSystem:
    """Система анализа TLE данных"""

    def __init__(self, model_path="model/tle_model.pth",
                 normalizer_path="model/normalizer.pkl",
                 device='cpu'):
        """
        Инициализация системы анализа

        Args:
            model_path: путь к обученной модели
            normalizer_path: путь к нормализатору
            device: устройство для вычислений
        """
        self.device = device

        # Загрузка модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Модель не найдена: {model_path}\n"
                f"Запустите train_model.py для обучения модели"
            )

        print(f"Загрузка модели из {model_path}...")
        self.model = TLEAnalyzer()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # Загрузка нормализатора
        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(
                f"Нормализатор не найден: {normalizer_path}\n"
                f"Запустите train_model.py для создания нормализатора"
            )

        print(f"Загрузка нормализатора из {normalizer_path}...")
        self.normalizer = TLENormalizer()
        self.normalizer.load(normalizer_path)

        # Порог для детекции аномалий (настраивается)
        self.anomaly_threshold = None
        self._calculate_threshold()

        print("✅ Система инициализирована успешно\n")

    def _calculate_threshold(self):
        """
        Вычисляет порог аномалии на основе статистики обучающих данных
        Можно улучшить, загрузив статистику из файла
        """
        # По умолчанию используем эвристический порог
        # В идеале нужно вычислить на validation set
        self.anomaly_threshold = 0.01  # будет настроен позже
        print(f"Порог детекции аномалий: {self.anomaly_threshold:.6f}")

    def analyze_single_tle(self, name_line, line1, line2, verbose=True):
        """
        Анализирует одну TLE запись

        Args:
            name_line: название объекта
            line1: первая строка TLE
            line2: вторая строка TLE
            verbose: выводить детальный анализ

        Returns:
            dict: результаты анализа
        """
        try:
            # Парсинг TLE
            vector, tle_struct = parse_tle(name_line, line1, line2)

            # Проверка типа орбиты
            mean_motion = tle_struct["mean_motion"]
            if not is_geo_orbit(mean_motion, tolerance=0.15):
                if verbose:
                    print(f"⚠️  {name_line}")
                    print(f"   Объект не на GEO орбите (mean_motion={mean_motion:.4f})")
                    print(f"   Рекомендации применимы только для GEO спутников\n")
                return {
                    "name": name_line.strip(),
                    "status": "not_geo",
                    "mean_motion": mean_motion
                }

            # Нормализация данных
            vector_normalized = self.normalizer.transform(vector)
            vector_tensor = torch.tensor(vector_normalized, dtype=torch.float32).unsqueeze(0)
            vector_tensor = vector_tensor.to(self.device)

            # Прогон через модель
            with torch.no_grad():
                reconstructed, latent = self.model(vector_tensor)
                reconstruction_error = torch.mean(
                    (vector_tensor - reconstructed) ** 2
                ).item()

                # Оценка аномальности
                anomaly_score = self.model.anomaly_detector(latent).item()

            # Денормализация предсказания
            reconstructed_denorm = self.normalizer.inverse_transform(
                reconstructed.cpu().numpy()[0]
            )

            # Детекция аномалии
            is_anomaly = reconstruction_error > self.anomaly_threshold

            # Генерация отчета
            if verbose:
                report = generate_detailed_text_with_values(
                    tle_struct,
                    reconstruction_error=reconstruction_error,
                    anomaly_score=anomaly_score,
                    threshold=self.anomaly_threshold
                )
                print(report)
                print()

            return {
                "name": name_line.strip(),
                "status": "analyzed",
                "tle_struct": tle_struct,
                "vector_original": vector,
                "vector_reconstructed": reconstructed_denorm.tolist(),
                "reconstruction_error": reconstruction_error,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "latent_representation": latent.cpu().numpy()[0].tolist()
            }

        except TLEParseError as e:
            if verbose:
                print(f"❌ Ошибка парсинга: {name_line}")
                print(f"   {str(e)}\n")
            return {
                "name": name_line.strip(),
                "status": "parse_error",
                "error": str(e)
            }
        except Exception as e:
            if verbose:
                print(f"❌ Непредвиденная ошибка: {name_line}")
                print(f"   {str(e)}\n")
            return {
                "name": name_line.strip(),
                "status": "error",
                "error": str(e)
            }

    def analyze_file(self, file_path, output_file=None, only_anomalies=False):
        """
        Анализирует файл с несколькими TLE записями

        Args:
            file_path: путь к файлу с TLE
            output_file: путь для сохранения отчета (опционально)
            only_anomalies: показывать только аномалии

        Returns:
            list: список результатов анализа
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        print("=" * 70)
        print(f"АНАЛИЗ ФАЙЛА: {file_path}")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()

        # Чтение файла
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        results = []

        # Обработка TLE записей
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                continue

            name_line = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            # Анализ
            result = self.analyze_single_tle(
                name_line, line1, line2,
                verbose=not only_anomalies
            )
            results.append(result)

            # Если режим only_anomalies, выводим только аномалии
            if only_anomalies and result.get('is_anomaly', False):
                report = generate_detailed_text_with_values(
                    result['tle_struct'],
                    reconstruction_error=result['reconstruction_error'],
                    anomaly_score=result['anomaly_score'],
                    threshold=self.anomaly_threshold
                )
                print(report)
                print()

        # Статистика
        print("\n" + "=" * 70)
        print("СТАТИСТИКА АНАЛИЗА")
        print("=" * 70)

        total = len(results)
        analyzed = sum(1 for r in results if r['status'] == 'analyzed')
        not_geo = sum(1 for r in results if r['status'] == 'not_geo')
        errors = sum(1 for r in results if r['status'] in ['parse_error', 'error'])
        anomalies = sum(1 for r in results if r.get('is_anomaly', False))

        print(f"Всего записей: {total}")
        print(f"  ✅ Проанализировано: {analyzed}")
        print(f"  ⚠️  Аномалий обнаружено: {anomalies}")
        print(f"  🌍 Не GEO орбиты: {not_geo}")
        print(f"  ❌ Ошибок: {errors}")

        if analyzed > 0:
            anomaly_rate = (anomalies / analyzed) * 100
            print(f"\nЧастота аномалий: {anomaly_rate:.1f}%")

        # Сохранение отчета
        if output_file:
            self._save_report(results, output_file)
            print(f"\n📄 Отчет сохранен в: {output_file}")

        print("=" * 70)

        return results

    def _save_report(self, results, output_file):
        """Сохраняет отчет в файл"""
        import json

        # Подготовка данных для JSON
        json_results = []
        for r in results:
            json_r = {k: v for k, v in r.items()
                      if k not in ['tle_struct']}  # исключаем сложные объекты

            if 'tle_struct' in r:
                tle = r['tle_struct']
                json_r['summary'] = generate_summary_text(tle)

            json_results.append(json_r)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_analyzed': len(results),
                'results': json_results
            }, f, indent=2, ensure_ascii=False)

    def compare_tle_epochs(self, tle_list):
        """
        Сравнивает несколько эпох одного спутника для отслеживания изменений

        Args:
            tle_list: список кортежей (name, line1, line2)

        Returns:
            dict: анализ изменений во времени
        """
        # TODO: реализовать анализ временных рядов
        pass


def main():
    """Главная функция с аргументами командной строки"""
    parser = argparse.ArgumentParser(
        description='Анализ TLE данных с использованием нейронной сети',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py --file data/test.txt
  python main.py --file data/test.txt --only-anomalies
  python main.py --file data/test.txt --output report.json
        """
    )

    parser.add_argument(
        '--file', '-f',
        type=str,
        default='data/test.txt',
        help='Путь к файлу с TLE данными (по умолчанию: data/test.txt)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default='model/tle_model.pth',
        help='Путь к модели (по умолчанию: model/tle_model.pth)'
    )

    parser.add_argument(
        '--normalizer', '-n',
        type=str,
        default='model/normalizer.pkl',
        help='Путь к нормализатору (по умолчанию: model/normalizer.pkl)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Путь для сохранения отчета в JSON'
    )

    parser.add_argument(
        '--only-anomalies', '-a',
        action='store_true',
        help='Показывать только аномалии'
    )

    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Принудительно использовать CPU'
    )

    args = parser.parse_args()

    # Определение устройства
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        # Инициализация системы
        system = TLEAnalysisSystem(
            model_path=args.model,
            normalizer_path=args.normalizer,
            device=device
        )

        # Анализ файла
        results = system.analyze_file(
            file_path=args.file,
            output_file=args.output,
            only_anomalies=args.only_anomalies
        )

    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА: {e}")
        print("\nУбедитесь, что:")
        print("  1. Обучена модель (запустите train_model.py)")
        print("  2. Существует файл с TLE данными")
        return 1
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
