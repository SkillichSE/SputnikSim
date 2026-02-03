"""
Главный скрипт для анализа TLE данных с учетом исторических трендов
"""
import torch
import numpy as np
from train_model import TLEAnalyzer, TLENormalizer
from parse_tle import parse_tle, TLEParseError, is_geo_orbit, ObjectType
from generate import generate_detailed_text_with_values, generate_summary_text
import os
import argparse
from datetime import datetime
import json
from collections import defaultdict


class TLEHistoryManager:
    """Управление историческими данными TLE"""
    def __init__(self, history_file="data/tle_history.json"):
        self.history_file = history_file
        self.history = defaultdict(list)  # catalog_number -> list of TLE structs
        self.load_history()
    
    def load_history(self):
        """Загружает историю из файла"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Конвертируем даты обратно
                    for cat_num, tle_list in data.items():
                        for tle in tle_list:
                            if 'epoch' in tle and 'datetime' in tle['epoch']:
                                tle['epoch']['datetime'] = datetime.fromisoformat(tle['epoch']['datetime'])
                        self.history[cat_num] = tle_list
                print(f"Загружена история для {len(self.history)} объектов")
            except Exception as e:
                print(f"Ошибка загрузки истории: {e}")
    
    def save_history(self):
        """Сохраняет историю в файл"""
        try:
            # Конвертируем datetime в ISO формат для JSON
            data_to_save = {}
            for cat_num, tle_list in self.history.items():
                data_to_save[cat_num] = []
                for tle in tle_list:
                    tle_copy = tle.copy()
                    if 'epoch' in tle_copy and 'datetime' in tle_copy['epoch']:
                        tle_copy['epoch'] = tle_copy['epoch'].copy()
                        tle_copy['epoch']['datetime'] = tle_copy['epoch']['datetime'].isoformat()
                    # Удаляем object_type и status_by_name (Enum не сериализуется)
                    tle_copy.pop('object_type', None)
                    tle_copy.pop('status_by_name', None)
                    data_to_save[cat_num].append(tle_copy)
            
            with open(self.history_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения истории: {e}")
    
    def add_tle(self, tle_struct):
        """Добавляет TLE в историю"""
        cat_num = tle_struct.get('catalog_number')
        if not cat_num:
            return
        
        # Добавляем в список
        self.history[cat_num].append(tle_struct)
        
        # Сортируем по времени
        self.history[cat_num] = sorted(
            self.history[cat_num],
            key=lambda x: x['epoch']['datetime']
        )
        
        # Ограничиваем до последних 10 записей
        if len(self.history[cat_num]) > 10:
            self.history[cat_num] = self.history[cat_num][-10:]
    
    def get_history(self, catalog_number):
        """Получает историю для объекта"""
        return self.history.get(catalog_number, [])


class TLEAnalysisSystem:
    """Система анализа TLE данных"""
    def __init__(self, model_path="model/tle_model.pth", 
                 normalizer_path="model/normalizer.pkl",
                 history_file="data/tle_history.json",
                 device='cpu'):
        self.device = device
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        print(f"Загрузка модели из {model_path}...")
        self.model = TLEAnalyzer()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        print(f"Загрузка нормализатора из {normalizer_path}...")
        self.normalizer = TLENormalizer()
        self.normalizer.load(normalizer_path)
        
        # История TLE
        self.history_manager = TLEHistoryManager(history_file)
        
        self.anomaly_threshold = 0.01
        print(f"Порог детекции аномалий: {self.anomaly_threshold:.6f}")
        print("✅ Система инициализирована\n")
    
    def analyze_single_tle(self, name_line, line1, line2, verbose=True, update_history=True):
        """Анализирует одну TLE запись с учетом истории"""
        try:
            vector, tle_struct = parse_tle(name_line, line1, line2)
            
            # Получаем историю
            cat_num = tle_struct.get('catalog_number')
            tle_history = self.history_manager.get_history(cat_num) if cat_num else []
            
            # Добавляем текущую запись в историю
            if update_history and cat_num:
                self.history_manager.add_tle(tle_struct)
            
            mean_motion = tle_struct["mean_motion"]
            if not is_geo_orbit(mean_motion, tolerance=0.15):
                if verbose:
                    print(f"⚠️  {name_line}")
                    print(f"   Объект не на GEO орбите (mean_motion={mean_motion:.4f})\n")
                return {"name": name_line.strip(), "status": "not_geo", "mean_motion": mean_motion}
            
            vector_normalized = self.normalizer.transform(vector)
            vector_tensor = torch.tensor(vector_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reconstructed, latent = self.model(vector_tensor)
                reconstruction_error = torch.mean((vector_tensor - reconstructed) ** 2).item()
                anomaly_score = self.model.anomaly_detector(latent).item()
            
            reconstructed_denorm = self.normalizer.inverse_transform(reconstructed.cpu().numpy()[0])
            is_anomaly = reconstruction_error > self.anomaly_threshold
            
            if verbose:
                report = generate_detailed_text_with_values(
                    tle_struct,
                    reconstruction_error=reconstruction_error,
                    anomaly_score=anomaly_score,
                    threshold=self.anomaly_threshold,
                    tle_history=tle_history,
                    reconstructed_vector=reconstructed_denorm  # Передаем восстановленный вектор!
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
                "history_count": len(tle_history)
            }

        except TLEParseError as e:
            if verbose:
                print(f"❌ Ошибка парсинга: {name_line}\n   {str(e)}\n")
            return {"name": name_line.strip(), "status": "parse_error", "error": str(e)}
        except Exception as e:
            if verbose:
                print(f"❌ Ошибка: {name_line}\n   {str(e)}\n")
            return {"name": name_line.strip(), "status": "error", "error": str(e)}

    def analyze_file(self, file_path, output_file=None, only_anomalies=False):
        """Анализирует файл с TLE записями"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        print("="*70)
        print(f"АНАЛИЗ ФАЙЛА: {file_path}")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        print()

        with open(file_path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        results = []

        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                continue
            name_line = lines[i]
            line1 = lines[i + 1]
            line2 = lines[i + 2]

            result = self.analyze_single_tle(name_line, line1, line2, verbose=not only_anomalies)
            results.append(result)

            if only_anomalies and result.get('is_anomaly', False):
                # Получаем восстановленный вектор из результата
                reconstructed = result.get('vector_reconstructed')

                report = generate_detailed_text_with_values(
                    result['tle_struct'],
                    reconstruction_error=result['reconstruction_error'],
                    anomaly_score=result['anomaly_score'],
                    threshold=self.anomaly_threshold,
                    tle_history=self.history_manager.get_history(result['tle_struct'].get('catalog_number')),
                    reconstructed_vector=reconstructed
                )
                print(report)
                print()
        
        # Сохраняем историю
        self.history_manager.save_history()
        
        # Статистика
        print("\n" + "="*70)
        print("СТАТИСТИКА АНАЛИЗА")
        print("="*70)
        
        total = len(results)
        analyzed = sum(1 for r in results if r['status'] == 'analyzed')
        not_geo = sum(1 for r in results if r['status'] == 'not_geo')
        errors = sum(1 for r in results if r['status'] in ['parse_error', 'error'])
        anomalies = sum(1 for r in results if r.get('is_anomaly', False))
        with_history = sum(1 for r in results if r.get('history_count', 0) > 0)
        
        print(f"Всего записей: {total}")
        print(f"  ✅ Проанализировано: {analyzed}")
        print(f"  ⚠️  Аномалий обнаружено: {anomalies}")
        print(f"  📊 С историческими данными: {with_history}")
        print(f"  🌍 Не GEO орбиты: {not_geo}")
        print(f"  ❌ Ошибок: {errors}")
        
        if analyzed > 0:
            anomaly_rate = (anomalies / analyzed) * 100
            print(f"\nЧастота аномалий: {anomaly_rate:.1f}%")
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_analyzed': len(results),
                    'anomalies': anomalies,
                    'results': [
                        {k: v for k, v in r.items() if k != 'tle_struct'}
                        for r in results
                    ]
                }, f, indent=2, ensure_ascii=False)
            print(f"\n📄 Отчет сохранен в: {output_file}")
        
        print("="*70)
        return results


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Анализ TLE данных с нейронной сетью')
    parser.add_argument('--file', '-f', type=str, default='data/test.txt')
    parser.add_argument('--model', '-m', type=str, default='model/tle_model.pth')
    parser.add_argument('--normalizer', '-n', type=str, default='model/normalizer.pkl')
    parser.add_argument('--history', type=str, default='data/tle_history.json', help='Файл истории TLE')
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--only-anomalies', '-a', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        system = TLEAnalysisSystem(
            model_path=args.model,
            normalizer_path=args.normalizer,
            history_file=args.history,
            device=device
        )
        system.analyze_file(file_path=args.file, output_file=args.output, only_anomalies=args.only_anomalies)
    except FileNotFoundError as e:
        print(f"\n❌ ОШИБКА: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
