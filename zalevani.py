#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64, json, os, sys, mimetypes, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from typing import Optional

from zoneinfo import ZoneInfo
from pydantic import BaseModel, ValidationError
from openai import OpenAI

# ---------- Hardcoded paths & settings ---------
IMAGE_PATH = "./plant.jpg"
WEATHER_NOW_PATH = "./weather_now.json"
WEATHER_FORECAST_PATH = "./weather_forecast.json"
HISTORY_PATH = "./watering_history.json"
PLANT_PATH = "./plant.json"
LOGFILE = "./watering.log"

ROTATE_MAX_MB = 5
ROTATE_BACKUPS = 3
MODEL = "gpt-4o-mini"
LOCAL_TZ = ZoneInfo("Europe/Prague")

# prompt log settings (w/o base64 picture)
LOG_PROMPT_INCLUDE_IMAGE = False

# ---------- Pydantic ----------
class WeatherNow(BaseModel):
    temp_c: float
    humidity_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    cloudcover_pct: Optional[float] = None
    description: Optional[str] = None

class WeatherForecastItem(BaseModel):
    time: datetime
    precip_prob_pct: Optional[float] = None
    expected_precip_mm: Optional[float] = None
    temp_c: Optional[float] = None
    humidity_pct: Optional[float] = None

class WeatherForecast(BaseModel):
    items: list[WeatherForecastItem]

class PlantContext(BaseModel):
    species: Optional[str] = None
    notes: Optional[str] = None

class DecisionRequest(BaseModel):
    image_url: str
    last_watering_date: Optional[str] = None
    last_watering_amount_ml: Optional[int] = None
    weather_now: WeatherNow
    weather_forecast: WeatherForecast
    plant: Optional[PlantContext] = None

class DecisionResponse(BaseModel):
    zalevat: bool
    oduvodneni: str

# ---------- Helpers ----------
def file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"Soubor '{path}' musí být obrázek (image/*).")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def load_json_or_default(path: str, default, required: bool = False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Soubor '{path}' neexistuje.")
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def days_since(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(LOCAL_TZ).date()
        return (datetime.now(LOCAL_TZ).date() - dt).days
    except Exception:
        return None

def precip_sum_next_hours(forecast: WeatherForecast, hours: int) -> float:
    now = datetime.now(LOCAL_TZ)
    end = now + timedelta(hours=hours)
    total = 0.0
    for it in forecast.items:
        t = it.time
        if t.tzinfo is None:
            t = t.replace(tzinfo=LOCAL_TZ)
        else:
            t = t.astimezone(LOCAL_TZ)
        if now <= t <= end and it.expected_precip_mm is not None:
            total += float(it.expected_precip_mm)
    return total

# ---------- Prompt ----------
def build_messages(payload: DecisionRequest) -> list:
    today_label = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")

    # tvrdá fakta pro model
    dsl = days_since(payload.last_watering_date)
    rain12 = precip_sum_next_hours(payload.weather_forecast, 12)
    facts = {
        "days_since_last_watering": dsl,
        "last_watering_amount_ml": payload.last_watering_amount_ml,
        "rain_next_12h_mm": round(rain12, 1),
        "now_temp_c": payload.weather_now.temp_c,
        "now_rh_pct": payload.weather_now.humidity_pct
    }

    system = (
        f"Dnes je {today_label} ráno. "
        "Rozhodni, zda **dnes** zalít rostlinu. POVINNÝ checklist: "
        "(1) fotka – známky přemokření/usušení, "
        "(2) DNY OD POSLEDNÍ ZÁLIVKY a POSLEDNÍ OBJEM (použij přesně tato FAKTA), "
        "(3) dnešní teplota a vlhkost (odpar), "
        "(4) srážky z předpovědi počasí na příštích 12 hodin. "
        "Pokud z faktů vyplývá horko (≥30 °C) a suchý vzduch (≤40 % RH) a poslední zálivka ≥1 den a déšť <2 mm/12 h, "
        "pak nezalévat lze jen s JASNÝM důvodem (např. velmi velká dávka včera, blízký výrazný déšť). "
        "Jinak zvol zalít. "
        "Odpověz **jen** jako JSON: "
        '{"zalevat": <true|false>, "oduvodneni": "<stručné, konkrétní"} '
        "Nesmíš si odporovat (např. horko+sucho bez brzkého deště a přesto nezalévat bez důvodu)."
    )

    user_text = (
        f"FAKTA: {facts}\n\n"
        f"Poslední zálivka (ISO): {payload.last_watering_date} "
        f"({payload.last_watering_amount_ml or 'neznámý'} ml)\n\n"
        f"Kontext rostliny: {payload.plant.model_dump_json() if payload.plant else 'neznámý'}\n\n"
        f"Aktuální počasí: {payload.weather_now.model_dump()}\n\n"
        f"Předpověď (prvních 12): {[i.model_dump() for i in payload.weather_forecast.items[:12]]}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": payload.image_url}},
        ]},
    ]

def serialize_messages_for_log(messages: list, include_image: bool) -> str:
    def scrub(o):
        if isinstance(o, dict):
            d = {}
            for k, v in o.items():
                if k == "image_url" and isinstance(v, dict) and "url" in v and not include_image:
                    d[k] = {"url": "<image-data-url-redacted>"}
                else:
                    d[k] = scrub(v)
            return d
        if isinstance(o, list): return [scrub(x) for x in o]
        return o
    return json.dumps(scrub(messages), ensure_ascii=False, indent=2)

# ---------- Open AI API Call ----------
def call_openai(messages: list, model: str) -> DecisionResponse:
    logging.info("PROMPT_MESSAGES:\n%s", serialize_messages_for_log(messages, LOG_PROMPT_INCLUDE_IMAGE))
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.0,
        response_format={"type": "json_object"}, max_tokens=400,
    )
    data = json.loads(resp.choices[0].message.content)
    return DecisionResponse(**data)

# ---------- Logging ----------
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    fh = RotatingFileHandler(LOGFILE, maxBytes=ROTATE_MAX_MB*1024*1024,
                             backupCount=ROTATE_BACKUPS, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.handlers = [sh, fh]

# ---------- Consistency Guard ----------
def consistency_guard(req: DecisionRequest, res: DecisionResponse) -> DecisionResponse:
    dsl = days_since(req.last_watering_date) or 999
    temp = req.weather_now.temp_c
    rh = req.weather_now.humidity_pct if req.weather_now.humidity_pct is not None else 100
    rain12 = precip_sum_next_hours(req.weather_forecast, 12)

    hot = temp >= 30.0
    dry = rh <= 40.0
    soon_rain = rain12 >= 2.0  # “výraznější” déšť do 12 h

    # Pokud je horko + sucho + min. 1 den od poslední zálivky + žádný výrazný déšť a model řekl NE,
    # přiklonit se k ANO s krátkým vysvětlením.
    if hot and dry and dsl >= 1 and not soon_rain and res.zalevat is False:
        res.zalevat = True
        res.oduvodneni = f"{dsl} d od zálivky, {temp:.0f}°C/{int(rh)}% RH, déšť <2 mm/12h → raději zalij."
    return res

# ---------- Main ----------
def main():
    setup_logging()

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("Chybí OPENAI_API_KEY v prostředí.")
        sys.exit(1)

    # picture → data URL
    try:
        image_data_url = file_to_data_url(IMAGE_PATH)
    except Exception as e:
        logging.error("Chyba při čtení obrázku: %s", e)
        sys.exit(1)

    # JSON inputs
    try:
        weather_now_raw = load_json_or_default(WEATHER_NOW_PATH, None, required=True)
        weather_forecast_raw = load_json_or_default(WEATHER_FORECAST_PATH, None, required=True)
        plant_raw = load_json_or_default(PLANT_PATH, None, required=False)
        history_raw = load_json_or_default(HISTORY_PATH, [], required=False)
    except Exception as e:
        logging.error("Chyba při načítání JSON: %s", e); sys.exit(1)

    # validation + “last watered”
    try:
        wn = WeatherNow(**weather_now_raw)
        wf = WeatherForecast(**weather_forecast_raw)
        plant = PlantContext(**plant_raw) if plant_raw else None

        last_date = None; last_amount = None
        if history_raw:
            last = max(history_raw, key=lambda e: e.get("date",""))
            last_date = last.get("date")
            last_amount = last.get("amount_ml")

        req = DecisionRequest(
            image_url=image_data_url,
            last_watering_date=last_date,
            last_watering_amount_ml=last_amount,
            weather_now=wn,
            weather_forecast=wf,
            plant=plant
        )
    except ValidationError as e:
        logging.error("Neplatná data: %s", e); sys.exit(1)

    messages = build_messages(req)

    try:
        res = call_openai(messages, model=MODEL)
    except Exception as e:
        logging.error("Chyba volání OpenAI: %s", e); sys.exit(1)

    # kconsistency_guard
    # res = consistency_guard(req, res)

    # output
    print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
    logging.info("RESPONSE: %s", json.dumps(res.model_dump(), ensure_ascii=False))

if __name__ == "__main__":
    main()
