# train_model.py
"""
Обучение модели для анализа и предсказания аномалий в TLE данных
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from parse_tle import parse_tle, is_geo_orbit
import os
import json
from sklearn.preprocessing import StandardScaler
import pickle


class TLEDataset(Dataset):
    """Dataset для TLE данных"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TLEAnalyzer(nn.Module):
    """
    Улучшенная модель для анализа TLE с автоэнкодером
    Обучается на нормальных GEO спутниках и детектирует аномалии
    """

    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=32):
        super().__init__()

        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, latent_dim),
            nn.ReLU()
        )

        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, input_dim)
        )

        # Классификатор аномалий (опционально)
        self.anomaly_detector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def detect_anomaly(self, x):
        """Детектирует аномалию по ошибке реконструкции"""
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            anomaly_score = self.anomaly_detector(latent)
        return mse, anomaly_score


class TLENormalizer:
    """Нормализация TLE параметров для лучшей работы модели"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, data):
        """Обучаем scaler на данных"""
        vectors = np.array([x for x, _ in data])
        self.scaler.fit(vectors)
        self.fitted = True

    def transform(self, vector):
        """Нормализуем вектор"""
        if not self.fitted:
            raise ValueError("Normalizer не обучен. Вызовите fit() сначала.")
        return self.scaler.transform(np.array(vector).reshape(1, -1))[0]

    def inverse_transform(self, vector):
        """Денормализуем вектор"""
        return self.scaler.inverse_transform(np.array(vector).reshape(1, -1))[0]

    def save(self, path):
        """Сохраняем scaler"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, path):
        """Загружаем scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True


def load_tle_dataset(file_path="data/all_tle.txt", geo_only=True):
    """
    Загружает TLE данные из файла

    Args:
        file_path: путь к файлу с TLE
        geo_only: загружать только GEO спутники

    Returns:
        list: список кортежей (vector, vector) для обучения
    """
    dataset = []
    errors = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден")

    with open(file_path, "r", encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Загружено {len(lines)} строк из {file_path}")

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            continue

        name_line = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        try:
            vector, tle_struct = parse_tle(name_line, line1, line2)

            # Фильтрация по типу орбиты
            if geo_only:
                mean_motion = tle_struct["mean_motion"]
                if not is_geo_orbit(mean_motion, tolerance=0.15):
                    continue

                # Дополнительные проверки для GEO
                if tle_struct["inclination"] > 15:  # GEO должен иметь малое наклонение
                    continue

            # Проверка на валидность данных
            if any(np.isnan(v) or np.isinf(v) for v in vector):
                errors.append(f"NaN/Inf в данных: {name_line}")
                continue

            dataset.append((vector, vector))  # Автоэнкодер: вход = выход

        except Exception as e:
            errors.append(f"Ошибка парсинга {name_line}: {str(e)}")

    print(f"\nУспешно загружено: {len(dataset)} TLE записей")
    if errors:
        print(f"Ошибок при парсинге: {len(errors)}")
        if len(errors) <= 10:
            for err in errors:
                print(f"  - {err}")

    return dataset


def train_model(model, dataset, normalizer, epochs=100, lr=0.001, batch_size=32,
                validation_split=0.2, device='cpu'):
    """
    Обучает модель с валидацией и early stopping

    Args:
        model: модель для обучения
        dataset: список данных
        normalizer: объект для нормализации
        epochs: количество эпох
        lr: learning rate
        batch_size: размер батча
        validation_split: доля данных для валидации
        device: устройство для обучения
    """
    model = model.to(device)

    # Нормализуем данные
    normalizer.fit(dataset)
    normalized_data = [(normalizer.transform(x), normalizer.transform(y))
                       for x, y in dataset]

    # Разделение на train/validation
    split_idx = int(len(normalized_data) * (1 - validation_split))
    train_data = normalized_data[:split_idx]
    val_data = normalized_data[split_idx:]

    print(f"\nРазделение данных:")
    print(f"  Train: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")

    # DataLoaders
    train_loader = DataLoader(TLEDataset(train_data), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(TLEDataset(val_data), batch_size=batch_size,
                            shuffle=False, drop_last=True)

    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=10)
    criterion = nn.MSELoss()

    # Early stopping
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': []
    }

    print("\n" + "=" * 60)
    print("Начало обучения")
    print("=" * 60)

    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            reconstructed, _ = model(x_batch)
            loss = criterion(reconstructed, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                reconstructed, _ = model(x_batch)
                loss = criterion(reconstructed, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Обновление learning rate
        scheduler.step(val_loss)

        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Вывод прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Сохранение лучшей модели
            if not os.path.exists("model"):
                os.makedirs("model")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, "model/tle_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping на эпохе {epoch + 1}")
                break

    print("\n" + "=" * 60)
    print("Обучение завершено")
    print(f"Лучший validation loss: {best_val_loss:.6f}")
    print("=" * 60)

    # Сохранение финальной модели
    torch.save(model.state_dict(), "model/tle_model.pth")
    normalizer.save("model/normalizer.pkl")

    # Сохранение истории
    with open("model/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nФайлы сохранены:")
    print("  - model/tle_model.pth (финальная модель)")
    print("  - model/tle_model_best.pth (лучшая модель)")
    print("  - model/normalizer.pkl (нормализатор)")
    print("  - model/training_history.json (история обучения)")

    return history


def evaluate_model(model, dataset, normalizer, device='cpu'):
    """Оценка качества модели на тестовых данных"""
    model.eval()
    model = model.to(device)

    # Нормализуем данные
    normalized_data = [(normalizer.transform(x), normalizer.transform(y))
                       for x, y in dataset]

    test_loader = DataLoader(TLEDataset(normalized_data), batch_size=32,
                             shuffle=False)

    reconstruction_errors = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            reconstructed, _ = model(x_batch)

            # Вычисляем ошибку реконструкции для каждого образца
            mse = torch.mean((x_batch - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    print("\n" + "=" * 60)
    print("Результаты оценки модели")
    print("=" * 60)
    print(f"Среднее значение ошибки реконструкции: {np.mean(reconstruction_errors):.6f}")
    print(f"Медиана ошибки реконструкции: {np.median(reconstruction_errors):.6f}")
    print(f"Стандартное отклонение: {np.std(reconstruction_errors):.6f}")
    print(f"95-й перцентиль: {np.percentile(reconstruction_errors, 95):.6f}")
    print("=" * 60)

    return reconstruction_errors


if __name__ == "__main__":
    # Проверка доступности CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Загрузка данных
    print("\nЗагрузка TLE данных...")
    dataset = load_tle_dataset("data/all_tle.txt", geo_only=True)

    if len(dataset) == 0:
        print("ОШИБКА: Не загружено ни одной записи. Проверьте файл data/all_tle.txt")
        exit(1)

    # Создание модели и нормализатора
    model = TLEAnalyzer(input_dim=6, hidden_dim=128, latent_dim=32)
    normalizer = TLENormalizer()

    # Обучение
    history = train_model(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        epochs=100,
        lr=0.001,
        batch_size=32,
        validation_split=0.2,
        device=device
    )

    # Оценка модели
    print("\nОценка модели на всех данных...")
    reconstruction_errors = evaluate_model(model, dataset, normalizer, device)
