# database.py (Исправленный с user_id в истории)
import aiosqlite
import json
import logging
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from typing import Any # Добавлено для type hinting

# --- Константы и соединение ---
load_dotenv()

try:
    # --- ИЗМЕНЕНО: Импортируем DB_FILE_PATH_ENV из bot_config ---
    from bot_config import DB_FILE_PATH_ENV
    DB_FILE = os.path.abspath(os.path.expanduser(DB_FILE_PATH_ENV))
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("bot_config not found, using default DB path.")
    DEFAULT_DB_PATH = "db/bot_db.sqlite"
    DB_FILE = os.path.abspath(os.path.expanduser(DEFAULT_DB_PATH))

logger = logging.getLogger(__name__)
logger.info(f"Database file path set to: {os.path.abspath(DB_FILE)}")

_connection: Optional[aiosqlite.Connection] = None
BUSY_TIMEOUT_MS = 5000

async def get_connection() -> aiosqlite.Connection:
    """Получает или создает асинхронное соединение с БД."""
    global _connection
    if _connection is None:
        try:
            db_dir = os.path.dirname(DB_FILE)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True) # Упрощено создание директории
                logger.info(f"Ensured database directory exists: {db_dir}")

            _connection = await aiosqlite.connect(
                DB_FILE,
                timeout=BUSY_TIMEOUT_MS / 1000.0
            )
            await _connection.execute("PRAGMA foreign_keys = ON")
            _connection.row_factory = aiosqlite.Row
            await _connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            await _connection.commit() # Commit PRAGMA statements
            logger.info(f"Database connection established to {DB_FILE} with busy_timeout={BUSY_TIMEOUT_MS}ms")
        except OSError as e:
             logger.critical(f"Failed to create/access database directory {db_dir}: {e}", exc_info=True)
             raise ConnectionError(f"Failed to create/access database directory: {e}") from e
        except Exception as e:
            logger.critical(f"Failed to connect to database {DB_FILE}: {e}", exc_info=True)
            raise ConnectionError(f"Failed to connect to database: {e}") from e
    return _connection

async def close_db():
    """Закрывает соединение с БД."""
    global _connection
    if _connection:
        try:
            await _connection.close()
            _connection = None
            logger.info("Database connection closed.")
        except Exception as e:
             logger.error(f"Error closing database connection: {e}", exc_info=True)


async def init_db():
    """Инициализирует таблицы в БД."""
    conn = await get_connection()
    try:
        # --- ИЗМЕНЕНО: Добавлено поле user_id ---
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'model')),
                user_id INTEGER NULL, -- ID пользователя для роли 'user', NULL для 'model'
                parts_json TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_chat_id ON chat_history (chat_id);')
        # Можно добавить индекс для user_id, если часто будете по нему искать
        # await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history (user_id);')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_notes (
                note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                category TEXT NOT NULL COLLATE NOCASE,
                value TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                UNIQUE (user_id, category)
            )
        ''')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_user_notes_user_id ON user_notes (user_id);')

        await conn.commit()
        logger.info("Database tables checked/created.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}", exc_info=True)
        # Можно рассмотреть вариант с rollback при ошибке
        # await conn.rollback()
        raise # Перевыбросить исключение, т.к. инициализация критична


# --- Функции для работы с историей ---

# _is_map_composite, _convert_value_for_json, _serialize_parts, _deserialize_parts - без изменений

def _is_map_composite(obj: Any) -> bool:
    """Проверяет, похож ли объект на MapComposite (утиная типизация)."""
    return hasattr(obj, 'keys') and hasattr(obj, 'values') and hasattr(obj, 'items') and \
           not isinstance(obj, dict)

def _convert_value_for_json(value: Any) -> Any:
    """Рекурсивно конвертирует значения для JSON-сериализации."""
    if isinstance(value, dict):
        return {str(k): _convert_value_for_json(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_convert_value_for_json(item) for item in value]
    elif _is_map_composite(value):
         logger.debug(f"Converting MapComposite-like object to dict: {type(value)}")
         return {str(k): _convert_value_for_json(v) for k, v in value.items()}
    elif hasattr(value, 'to_dict') and callable(value.to_dict):
        try:
            dict_repr = value.to_dict()
            return _convert_value_for_json(dict_repr)
        except Exception as e:
            logger.warning(f"Calling to_dict() failed for {type(value)}: {e}. Converting to string.")
            return str(value)
    elif isinstance(value, (str, int, float, bool, type(None))):
        return value
    else:
        logger.warning(f"Cannot directly serialize type {type(value)}. Converting to string.")
        return str(value)

def _serialize_parts(parts: List[Any]) -> str:
    """Сериализует список Parts в JSON строку."""
    serializable_list = []
    if not isinstance(parts, list):
        logger.warning(f"Input to _serialize_parts is not a list: {type(parts)}. Returning empty list.")
        return "[]"
    for item in parts:
        converted_item = _convert_value_for_json(item)
        if converted_item is not None:
            serializable_list.append(converted_item)
        else:
             logger.warning(f"Skipping item after conversion resulted in None: {item}")
    try:
        return json.dumps(serializable_list, ensure_ascii=False, indent=None)
    except TypeError as e:
        logger.error(f"JSON serialization failed after conversion: {e}. Data sample: {str(serializable_list)[:500]}...", exc_info=True)
        return "[]"
    except Exception as e:
        logger.error(f"Unexpected error during JSON serialization: {e}", exc_info=True)
        return "[]"

def _deserialize_parts(parts_json: str) -> List[Dict[str, Any]]:
    """Десериализует JSON строку обратно в список словарей."""
    if not isinstance(parts_json, str):
        logger.error(f"Input to _deserialize_parts is not a string: {type(parts_json)}")
        return []
    try:
        data = json.loads(parts_json)
        if isinstance(data, list):
            return data
        else:
            logger.error(f"Deserialized JSON is not a list: {type(data)}")
            return []
    except json.JSONDecodeError:
        logger.error(f"Failed to deserialize parts_json: {parts_json[:500]}...")
        return []

# --- ИЗМЕНЕНО: Добавлен параметр user_id и он используется в INSERT ---
async def add_message_to_history(chat_id: int, role: str, parts: List[Any], user_id: Optional[int] = None):
    """Добавляет сообщение в историю чата, включая user_id для сообщений пользователя."""
    if role not in ('user', 'model'):
        logger.error(f"Invalid role '{role}' for chat history.")
        return
    if not parts:
        logger.debug(f"Attempted to add empty parts to history for chat_id {chat_id}, role {role}. Skipping.")
        return

    # Для сообщений модели user_id должен быть None
    if role == 'model' and user_id is not None:
        logger.warning(f"Forcing user_id to NULL for role 'model' in chat {chat_id}.")
        user_id = None
    # Для сообщений пользователя user_id должен быть указан
    elif role == 'user' and user_id is None:
        logger.error(f"Missing user_id for role 'user' in chat {chat_id}. History entry might be incomplete.")
        # Решите, хотите ли вы прерывать сохранение или сохранять с NULL
        # return

    parts_json = _serialize_parts(parts)
    if parts_json == "[]" and parts:
         logger.error(f"Serialization failed for non-empty parts for chat {chat_id}, role {role}. History not saved.")
         return

    conn = await get_connection()
    try:
        await conn.execute(
            "INSERT INTO chat_history (chat_id, role, user_id, parts_json) VALUES (?, ?, ?, ?)", # Добавлен user_id
            (chat_id, role, user_id, parts_json) # Передаем user_id
        )
        await conn.commit()
        logger.debug(f"Added message to history: chat_id={chat_id}, role={role}, user_id={user_id}, json_size={len(parts_json)}")
    except Exception as e:
        logger.error(f"Failed to insert message into history for chat_id {chat_id}: {e}", exc_info=True)
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


# --- ИЗМЕНЕНО: Добавлен SELECT user_id и он возвращается в словаре ---
async def get_chat_history(chat_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Получает историю чата из БД, включая user_id для сообщений пользователя.
    Возвращает список словарей: [{role: ..., user_id: ... (опционально), parts: [...]}, ...].
    """
    conn = await get_connection()
    gemini_history = [] # Инициализируем здесь на случай ошибки
    try:
        cursor = await conn.execute(
            "SELECT role, user_id, parts_json FROM chat_history WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?", # Добавлен user_id
            (chat_id, limit)
        )
        rows = await cursor.fetchall()
        await cursor.close()
    except Exception as e:
        logger.error(f"Failed to fetch chat history for chat_id {chat_id}: {e}", exc_info=True)
        return [] # Возвращаем пустой список при ошибке

    for row in reversed(rows): # Идем от старых к новым
        parts_list = _deserialize_parts(row['parts_json'])
        if parts_list:
             entry = {"role": row["role"], "parts": parts_list}
             # Добавляем user_id, если это запись пользователя и ID не NULL
             if row["role"] == 'user' and row['user_id'] is not None:
                  entry['user_id'] = row['user_id']
             gemini_history.append(entry)

    logger.debug(f"Retrieved {len(gemini_history)} history entries for chat_id={chat_id}")
    return gemini_history
# --- КОНЕЦ ИЗМЕНЕНИЯ ---


# --- Функции для работы с профилями и заметками (без изменений) ---
async def upsert_user_profile(user_id: int, username: Optional[str], first_name: Optional[str], last_name: Optional[str]):
    """Добавляет или обновляет базовую информацию о пользователе."""
    conn = await get_connection()
    try:
        await conn.execute('''
            INSERT INTO user_profiles (user_id, username, first_name, last_name, last_seen)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_name = excluded.last_name,
                last_seen = CURRENT_TIMESTAMP
        ''', (user_id, username, first_name, last_name))
        await conn.commit()
        logger.debug(f"Upserted user profile for user_id {user_id}") # Изменен лог на debug
    except Exception as e:
         logger.error(f"Failed to upsert user profile for user_id {user_id}: {e}", exc_info=True)


async def upsert_user_note(user_id: int, category: str, value: str) -> bool:
    """Добавляет или обновляет заметку о пользователе."""
    if not category or not value:
        logging.warning(f"Attempted to save empty category or value for user_id {user_id}")
        return False
    conn = await get_connection()
    cursor = None
    try:
        cursor = await conn.execute('''
            INSERT INTO user_notes (user_id, category, value, timestamp)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, category) DO UPDATE SET
                value = excluded.value,
                timestamp = CURRENT_TIMESTAMP
        ''', (user_id, category.strip().lower(), value.strip()))
        await conn.commit()
        # rowcount может быть ненадежен, логируем успех
        logging.info(f"Upserted note for user_id={user_id}: category='{category.strip().lower()}'")
        return True
    except Exception as e:
        logger.error(f"Error upserting user note for user_id={user_id}, category='{category}': {e}", exc_info=True)
        return False
    finally:
        if cursor:
            try: await cursor.close()
            except Exception as cur_e: logger.error(f"Error closing cursor during note upsert: {cur_e}")


async def get_user_notes(user_id: int) -> Dict[str, str]:
    """Получает все заметки о пользователе в виде словаря {категория: значение}."""
    conn = await get_connection()
    cursor = None
    notes = {} # Инициализируем здесь
    try:
        cursor = await conn.execute(
            "SELECT category, value FROM user_notes WHERE user_id = ? ORDER BY category",
            (user_id,)
        )
        rows = await cursor.fetchall()
        notes = {row['category']: row['value'] for row in rows}
        logger.debug(f"Retrieved {len(notes)} notes for user_id={user_id}")
    except Exception as e:
        logger.error(f"Error getting user notes for user_id {user_id}: {e}", exc_info=True)
    finally:
         if cursor:
             try: await cursor.close()
             except Exception as cur_e: logger.error(f"Error closing cursor during note get: {cur_e}")
    return notes # Возвращаем notes (может быть пустым словарем)


async def delete_user_note(user_id: int, category: str) -> bool:
     """Удаляет заметку пользователя по категории (без учета регистра)."""
     conn = await get_connection()
     cursor = None
     success = False # Инициализируем
     try:
          cursor = await conn.execute(
               "DELETE FROM user_notes WHERE user_id = ? AND lower(category) = ?",
               (user_id, category.strip().lower())
          )
          await conn.commit()
          deleted_count = cursor.rowcount
          if deleted_count > 0:
               logger.info(f"Deleted note for user_id={user_id}, category='{category.strip().lower()}'")
               success = True
          else:
               logger.info(f"Note not found for deletion: user_id={user_id}, category='{category.strip().lower()}'")
               success = False # Явно указываем
     except Exception as e:
          logger.error(f"Error deleting user note for user_id={user_id}, category='{category}': {e}", exc_info=True)
          success = False
     finally:
          if cursor:
              try: await cursor.close()
              except Exception as cur_e: logger.error(f"Error closing cursor during note delete: {cur_e}")
     return success