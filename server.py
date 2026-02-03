# server.py
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from train_model import TLEAnalyzer
from generate import generate_detailed_text_with_values
from pyngrok import ngrok

# === Ngrok ===
ngrok.set_auth_token("38R6aPY9h028XFaGm0qS8LLRHli_81F5JiNGKdk3yc6pUaVgL")

# === FastAPI ===
app = FastAPI(title="TLE Analyzer API")

class TLERequest(BaseModel):
    tle_lines: list[str]

# === Загружаем модель ===
MODEL_PATH = "model/tle_model.pth"
model = TLEAnalyzer()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

def split_tle_line(tle_text: str) -> list[str]:
    parts = tle_text.strip().split()
    if len(parts) < 23:
        raise ValueError("Недостаточно полей для TLE")

    name = parts[0]
    line1 = " ".join(parts[1:8])
    line2 = " ".join(parts[8:])

    return [name, line1, line2]

# === Безопасный парсинг TLE ===
def parse_tle_safe(name_line, line1, line2):
    # Приведение к строкам на всякий случай
    name_line = str(name_line)
    line1 = str(line1)
    line2 = str(line2)

    def parse_float(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0  # 0 вместо None
    def parse_int(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return 0

    tle_struct = {
        "name": name_line.strip(),
        "line1": line1.strip(),
        "line2": line2.strip(),
        "year": parse_int(line1[18:20].strip()) if len(line1) >= 20 else 0,
        "inclination": parse_float(line2[8:16].strip()) if len(line2) >= 16 else 0.0,
        "raan": parse_float(line2[17:25].strip()) if len(line2) >= 25 else 0.0,
        "eccentricity": parse_float(line2[26:33].strip()) if len(line2) >= 33 else 0.0,
        "argument_perigee": parse_float(line2[34:42].strip()) if len(line2) >= 42 else 0.0,
        "mean_anomaly": parse_float(line2[43:51].strip()) if len(line2) >= 51 else 0.0,
        "mean_motion": parse_float(line2[52:63].strip()) if len(line2) >= 63 else 0.0,
    }

    vector = [
        tle_struct["inclination"],
        tle_struct["raan"],
        tle_struct["eccentricity"],
        tle_struct["argument_perigee"],
        tle_struct["mean_anomaly"],
        tle_struct["mean_motion"],
    ]

    return vector, tle_struct

# === Эндпоинт ===
@app.post("/analyze/")
async def analyze_tle(request: TLERequest):
    if len(request.tle_lines) != 3:
        return JSONResponse(status_code=400, content={"error": "Нужно передать 3 строки: имя + line1 + line2"})

    name_line, line1, line2 = request.tle_lines

    try:
        vector, tle_struct = parse_tle_safe(name_line, line1, line2)

        # Модель предсказания
        x = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_vector = model(x)[0].tolist()

        # Генерация текста с проверкой
        try:
            text = generate_detailed_text_with_values(tle_struct, predicted_vector=pred_vector)
        except Exception as e:
            text = f"[Ошибка генерации текста]: {e}"

        return {"recommendations": text}

    except Exception as e:
        return JSONResponse(status_code=200, content={"error": f"Ошибка обработки TLE: {e}"})

# Запуск через Ngrok
if __name__ == "__main__":
    public_url = ngrok.connect(8000)
    print(f"API доступен в интернете: {public_url}")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
