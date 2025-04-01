# bot_lifecycle.py
import json
import logging
import inspect

# --- Локальные импорты ---
import database
# --- ИЗМЕНЕНИЕ: Импортируем переименованный файл ---
import tool_handlers # <--- Импортируем файл с функциями-инструментами
from gemini_api import setup_gemini_model
from bot_config import (
    GOOGLE_API_KEY,
    LITE_GEMINI_MODEL_NAME, PRO_GEMINI_MODEL_NAME,
    LITE_FUNC_DECL_FILE, PRO_FUNC_DECL_FILE,
    LITE_PROMPT_FILE, PRO_PROMPT_FILE,
    MAX_LITE_MODEL_FC_STEPS, MAX_PRO_MODEL_FC_STEPS
)
from bot_loader import dp

logger = logging.getLogger(__name__)

async def on_startup():
    """Инициализация ресурсов при старте бота."""
    local_lite_model = None
    local_pro_model = None
    local_lite_function_declarations = []
    local_pro_function_declarations = []
    local_lite_system_prompt = None
    local_pro_system_prompt = None
    local_available_pro_functions = {}
    local_available_lite_functions = {}

    logger.info("Executing bot startup sequence...")

    # 1. Инициализация БД (без изменений)
    try:
        await database.init_db(); logger.info("Database initialized.")
    except Exception as db_init_err: logger.critical(f"FATAL: DB init failed: {db_init_err}", exc_info=True); exit(1)

    # 2. Загрузка деклараций функций (без изменений)
    try:
        with open(LITE_FUNC_DECL_FILE, "r", encoding="utf-8") as f: local_lite_function_declarations = json.load(f)
        logger.info(f"Loaded {len(local_lite_function_declarations)} Lite func decls from {LITE_FUNC_DECL_FILE}.")
    except Exception as e: logger.error(f"Failed loading {LITE_FUNC_DECL_FILE}: {e}"); local_lite_function_declarations = []
    try:
        with open(PRO_FUNC_DECL_FILE, "r", encoding="utf-8") as f: local_pro_function_declarations = json.load(f)
        logger.info(f"Loaded {len(local_pro_function_declarations)} Pro func decls from {PRO_FUNC_DECL_FILE}.")
    except Exception as e: logger.error(f"Failed loading {PRO_FUNC_DECL_FILE}: {e}"); local_pro_function_declarations = []

    # 3. Загрузка системных промптов (без изменений)
    try:
        with open(LITE_PROMPT_FILE, "r", encoding="utf-8") as f: local_lite_system_prompt = f.read()
        logger.info(f"Lite system prompt loaded from {LITE_PROMPT_FILE}.")
    except Exception as e: logger.error(f"Failed loading {LITE_PROMPT_FILE}: {e}"); local_lite_system_prompt = None
    try:
        with open(PRO_PROMPT_FILE, "r", encoding="utf-8") as f: local_pro_system_prompt = f.read()
        logger.info(f"Pro system prompt loaded from {PRO_PROMPT_FILE}.")
    except Exception as e: logger.error(f"Failed loading {PRO_PROMPT_FILE}: {e}"); local_pro_system_prompt = None

    # 4. Инициализация моделей Gemini (без изменений)
    try:
        local_lite_model = setup_gemini_model(GOOGLE_API_KEY, local_lite_function_declarations, LITE_GEMINI_MODEL_NAME, local_lite_system_prompt)
        if local_lite_model: logger.info(f"Lite model '{LITE_GEMINI_MODEL_NAME}' initialized.")
        else: raise ValueError("setup_gemini_model returned None for Lite")
    except Exception as e: logger.critical(f"FATAL: Lite model init fail: {e}", exc_info=True); exit(1)
    try:
        local_pro_model = setup_gemini_model(GOOGLE_API_KEY, local_pro_function_declarations, PRO_GEMINI_MODEL_NAME, local_pro_system_prompt)
        if local_pro_model: logger.info(f"Pro model '{PRO_GEMINI_MODEL_NAME}' initialized.")
        else: raise ValueError("setup_gemini_model returned None for Pro")
    except Exception as e: logger.critical(f"FATAL: Pro model init fail: {e}", exc_info=True); exit(1)

    # 5. Маппинг хендлеров функций
    # --- ИЗМЕНЕНИЕ: Ищем функции в модуле tool_handlers ---
    all_handler_names = [name for name, func in inspect.getmembers(tool_handlers, inspect.isfunction)]
    # --------------------------------------------------------
    pro_func_names = [
        "remember_user_info", "get_current_weather", "get_stock_price", "read_file_from_env",
        "write_file_to_env", "execute_python_script_in_env", "create_file_in_env",
        "execute_terminal_command_in_env", "edit_file_content", "replace_code_block_ast",
        "edit_json_file", "send_telegram_message"
    ]
    # --- ИЗМЕНЕНИЕ: Получаем атрибут из tool_handlers ---
    local_available_pro_functions = {name: getattr(tool_handlers, name) for name in pro_func_names if name in all_handler_names}
    # --------------------------------------------------
    missing_pro = [name for name in pro_func_names if name not in local_available_pro_functions]
    if missing_pro: logger.warning(f"Handlers not found for Pro funcs: {missing_pro}")
    logger.info(f"Mapped {len(local_available_pro_functions)} handlers for Pro model.")

    lite_func_names = ["remember_user_info"] # Lite модель может вызвать trigger_pro_model_processing, но сам хендлер не нужен в Lite мапе
    # --- ИЗМЕНЕНИЕ: Получаем атрибут из tool_handlers ---
    local_available_lite_functions = {name: getattr(tool_handlers, name) for name in lite_func_names if name in all_handler_names}
    # --------------------------------------------------
    missing_lite = [name for name in lite_func_names if name not in local_available_lite_functions]
    if missing_lite: logger.warning(f"Handlers not found for Lite funcs: {missing_lite}")
    logger.info(f"Mapped {len(local_available_lite_functions)} handlers for Lite model.")

    # 6. Сохраняем данные в dp.workflow_data (без изменений)
    dp.workflow_data.update({
        "lite_model": local_lite_model,
        "pro_model": local_pro_model,
        "available_lite_functions": local_available_lite_functions,
        "available_pro_functions": local_available_pro_functions,
        "max_lite_steps": MAX_LITE_MODEL_FC_STEPS,
        "max_pro_steps": MAX_PRO_MODEL_FC_STEPS
    })
    logger.info("Models and function maps added to Dispatcher workflow_data.")

    logger.info("Bot startup sequence complete!")


async def on_shutdown():
    """Действия при остановке бота."""
    logger.info("Executing bot shutdown sequence...")
    try: await database.close_db(); logger.info("Database connection closed.")
    except Exception as e: logger.error(f"Error closing database: {e}", exc_info=True)
    logger.info("Bot shutdown sequence complete.")