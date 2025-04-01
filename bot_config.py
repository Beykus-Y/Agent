# bot_config.py
import os
import logging
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# --- Основные Токены и Ключи ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Названия моделей Gemini ---
LITE_GEMINI_MODEL_NAME = os.getenv("LITE_GEMINI_MODEL", "gemini-1.5-flash-latest")
PRO_GEMINI_MODEL_NAME = os.getenv("PRO_GEMINI_MODEL", "gemini-1.5-pro-latest")

# --- Пути к файлам ---
LITE_FUNC_DECL_FILE = "lite_function_declarations.json"
PRO_FUNC_DECL_FILE = "pro_function_declarations.json" # Используем один для Pro
LITE_PROMPT_FILE = "prompt/lite_model_group_analyzer.txt"
PRO_PROMPT_FILE = "prompt/pro_model_assistant.txt"
DB_FILE_PATH_ENV = os.getenv("DATABASE_FILE_PATH", "db/bot_db.sqlite") # Путь к БД из .env или по умолчанию

# --- Константы обработки ---
MAX_PRO_MODEL_FC_STEPS = 10 # Макс. шагов обработки FC для Pro-модели
MAX_LITE_MODEL_FC_STEPS = 3 # Макс. шагов обработки FC для Lite-модели

# --- Проверка наличия обязательных переменных ---
if not BOT_TOKEN:
    logging.critical("FATAL: TELEGRAM_BOT_TOKEN not found in .env")
    exit(1)
if not GOOGLE_API_KEY:
    logging.critical("FATAL: GOOGLE_API_KEY not found in .env")
    exit(1)

# --- Настройка логирования ---
# Вынесена сюда для централизации
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("Configuration loaded.")
logger.info(f"Lite Model: {LITE_GEMINI_MODEL_NAME}, Pro Model: {PRO_GEMINI_MODEL_NAME}")