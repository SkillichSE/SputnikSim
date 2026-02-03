"""
Обучение модели для анализа и предсказания аномалий в TLE данных
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from parse_tle import parse_tle, is_geo_orbit, ObjectType
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
    """Автоэнкодер для детекции аномалий в TLE"""
    def __init__(self, input_dim=6, hidden_dim=128, latent_dim=32):
        super().__init__()
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
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            anomaly_score = self.anomaly_detector(latent)
        return mse, anomaly_score


class TLENormalizer:
    """Нормализация TLE параметров"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    def fit(self, data):
        vectors = np.array([x for x, _ in data])
        self.scaler.fit(vectors)
        self.fitted = True
    def transform(self, vector):
        if not self.fitted:
            raise ValueError("Normalizer не обучен")
        return self.scaler.transform(np.array(vector).reshape(1, -1))[0]
    def inverse_transform(self, vector):
        return self.scaler.inverse_transform(np.array(vector).reshape(1, -1))[0]
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    def load(self, path):
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True


def load_tle_dataset(file_path="data/all_tle.txt", geo_only=True, exclude_debris=True):
    """Загружает TLE данные из файла"""
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
            
            # Исключаем мусор при обучении обучаем только на спутниках
            if exclude_debris:
                obj_type = tle_struct.get("object_type", ObjectType.UNKNOWN)
                if obj_type in [ObjectType.DEBRIS, ObjectType.FRAGMENT, ObjectType.ROCKET_BODY]:
                    continue
            
            if geo_only:
                mean_motion = tle_struct["mean_motion"]
                if not is_geo_orbit(mean_motion, tolerance=0.15):
                    continue
                if tle_struct["inclination"] > 15:
                    continue
            
            if any(np.isnan(v) or np.isinf(v) for v in vector):
                errors.append(f"NaN/Inf в данных: {name_line}")
                continue
            
            dataset.append((vector, vector))
            
        except Exception as e:
            errors.append(f"Ошибка парсинга {name_line}: {str(e)}")
    
    print(f"\nУспешно загружено: {len(dataset)} TLE записей")
    if errors:
        print(f"Ошибок при парсинге: {len(errors)}")
    
    return dataset


def train_model(model, dataset, normalizer, epochs=100, lr=0.001, batch_size=32, 
                validation_split=0.2, device='cpu'):
    """Обучает модель"""
    model = model.to(device)
    normalizer.fit(dataset)
    normalized_data = [(normalizer.transform(x), normalizer.transform(y)) for x, y in dataset]
    
    split_idx = int(len(normalized_data) * (1 - validation_split))
    train_data = normalized_data[:split_idx]
    val_data = normalized_data[split_idx:]
    
    print(f"\nРазделение: Train={len(train_data)}, Val={len(val_data)}")
    
    train_loader = DataLoader(TLEDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TLEDataset(val_data), batch_size=batch_size, shuffle=False, drop_last=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    print("\n" + "="*60)
    print("Начало обучения")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(x_batch)
            loss = criterion(reconstructed, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                reconstructed, _ = model(x_batch)
                loss = criterion(reconstructed, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if not os.path.exists("model"):
                os.makedirs("model")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'train_loss': train_loss, 'val_loss': val_loss,
                       }, "model/tle_model_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping на эпохе {epoch+1}")
                break
    
    print(f"\nЛучший validation loss: {best_val_loss:.6f}")
    torch.save(model.state_dict(), "model/tle_model.pth")
    normalizer.save("model/normalizer.pkl")
    with open("model/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Модели сохранены")
    return history


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    dataset = load_tle_dataset("data/all_tle.txt", geo_only=True, exclude_debris=True)
    if len(dataset) == 0:
        print("ОШИБКА: Нет данных")
        exit(1)
    model = TLEAnalyzer(input_dim=6, hidden_dim=128, latent_dim=32)
    normalizer = TLENormalizer()
    history = train_model(model, dataset, normalizer, epochs=100, lr=0.001, batch_size=32, validation_split=0.2, device=device)
