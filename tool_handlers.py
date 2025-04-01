# ./handlers.py
import os
import subprocess
import time
import json
import ast
from pathlib import Path
import logging
import asyncio
from typing import Optional, List, Dict, Any, Tuple, Union
from bot_loader import bot # Нужен экземпляр бота для отправки
from bot_utils import escape_markdown_v2 # Для экранирования
# --- Ваши модули ---
import database # Модуль для работы с БД
import aiofiles # Для асинхронных файловых операций
# --- ИСПРАВЛЕНО: Добавляем правильный импорт для асинхронных операций с путями ---
import aiofiles.os

# --- Константы ---
ENV_DIR = "env" # Корневая директория для окружений чатов
ABS_ENV_DIR = os.path.abspath(ENV_DIR) # Абсолютный путь к корневой директории окружений

logger = logging.getLogger(__name__) # Именной логгер для этого модуля

# --- Вспомогательные функции для работы с путями и директориями ---

async def send_telegram_message(
    chat_id: int, # Добавляем chat_id, т.к. бот должен знать КУДА отправлять
    text: str,
    delay_seconds: int = 0,
    # expect_response: bool = False # Этот параметр сложно реализовать напрямую через FC
    ) -> dict:
    """
    Отправляет текстовое сообщение в указанный чат Telegram. (Async)
    """
    logger.info(f"--- Async Function Call: send_telegram_message(chat_id={chat_id}, text='{text[:50]}...', delay={delay_seconds}) ---")
    if not text:
        logger.warning(f"Attempted send_telegram_message with empty text to chat {chat_id}")
        return {"status": "error", "message": "Cannot send empty message."}

    try:
        if delay_seconds > 0:
            logger.info(f"Delaying message send by {delay_seconds} seconds for chat {chat_id}")
            await asyncio.sleep(delay_seconds)

        escaped_text = escape_markdown_v2(text)
        await bot.send_message(chat_id=chat_id, text=escaped_text)
        logger.info(f"Successfully sent message via send_telegram_message to chat {chat_id}")
        # ВАЖНО: Эта функция просто отправляет сообщение.
        # Она НЕ заменяет собой финальный ответ модели.
        # Модель должна после вызова этой функции сгенерировать свой финальный ответ
        # (который может быть просто "Сообщение отправлено" или пустой, если больше сказать нечего).
        return {"status": "success", "message": "Message sent successfully."}
    except Exception as e:
        logger.error(f"Error sending message via send_telegram_message to chat {chat_id}: {e}", exc_info=True)
        # Возвращаем ошибку, чтобы модель знала о проблеме
        return {"status": "error", "message": f"Failed to send message: {e}"}

def _get_safe_chat_path(chat_id: int, filename: str) -> tuple[bool, Optional[str]]:
    """
    Строит и проверяет путь к файлу внутри ИЗОЛИРОВАННОЙ директории чата.
    Возвращает (is_safe, absolute_path) или (False, None).
    НЕ СОЗДАЕТ ДИРЕКТОРИИ ЗДЕСЬ.
    """
    if not isinstance(chat_id, int):
        logger.error(f"Invalid chat_id type for path: {chat_id}")
        return False, None
    # Проверяем filename на пустоту, тип и попытки выхода из директории ('..')
    if not filename or not isinstance(filename, str) or '..' in filename.replace('\\', '/').split('/'):
         logger.error(f"Invalid or potentially unsafe filename for chat {chat_id}: '{filename}'")
         return False, None

    try:
        chat_dir_rel = str(chat_id) # Относительный путь директории чата
        chat_dir_abs = os.path.abspath(os.path.join(ABS_ENV_DIR, chat_dir_rel))

        # Нормализуем имя файла, убирая начальные слеши, если они есть
        normalized_filename = filename.lstrip('/\\')

        # Формируем полный предполагаемый путь к файлу
        filepath_abs = os.path.abspath(os.path.join(chat_dir_abs, normalized_filename))

        # Ключевая проверка безопасности: путь должен начинаться с директории чата
        if filepath_abs.startswith(chat_dir_abs + os.sep) or filepath_abs == chat_dir_abs:
            return True, filepath_abs
        else:
            logger.warning(f"Path traversal attempt denied for chat {chat_id}: filename='{filename}', resolved='{filepath_abs}', expected_prefix='{chat_dir_abs + os.sep}'")
            return False, None
    except Exception as e:
        logger.error(f"Error checking path safety for chat {chat_id}, file '{filename}': {e}", exc_info=True)
        return False, None

def _ensure_chat_dir_exists(chat_id: int):
     """Синхронно создает директорию чата внутри ABS_ENV_DIR, если ее нет."""
     if not isinstance(chat_id, int):
         logger.error(f"Cannot create directory for invalid chat_id: {chat_id}")
         return
     try:
         chat_dir = os.path.join(ABS_ENV_DIR, str(chat_id))
         os.makedirs(chat_dir, exist_ok=True)
     except Exception as e:
          logger.error(f"Failed to create chat directory for chat_id {chat_id}: {e}", exc_info=True)
          raise IOError(f"Failed to create environment for chat {chat_id}") from e

# --- Хендлеры функций для Gemini ---

async def remember_user_info(user_id: int, info_category: str, info_value: str) -> dict:
    """
    Сохраняет или обновляет заметку о пользователе в БД. (Async)
    Возвращает словарь со статусом операции.
    """
    logger.info(f"--- Function Call: remember_user_info(user_id={user_id}, category='{info_category}', value='{info_value[:50]}...') ---")
    if not isinstance(user_id, int) or user_id <= 0:
         logger.error(f"Invalid user_id provided: {user_id}")
         return {"status": "error", "message": "Invalid user_id provided."}
    if not info_category or not isinstance(info_category, str):
         logger.error(f"Invalid info_category provided for user_id {user_id}: {info_category}")
         return {"status": "error", "message": "Invalid or empty info_category provided."}
    if not info_value or not isinstance(info_value, str):
         logger.error(f"Invalid info_value provided for user_id {user_id}, category {info_category}: {info_value}")
         return {"status": "error", "message": "Invalid or empty info_value provided."}
    try:
        success = await database.upsert_user_note(user_id, info_category.strip().lower(), info_value.strip())
        if success:
            return {"status": "success", "message": f"Information saved for user {user_id} under category '{info_category.strip().lower()}'."}
        else:
            return {"status": "error", "message": "Failed to save information to the database."}
    except Exception as e:
        logger.error(f"Unexpected error in remember_user_info handler: {e}", exc_info=True)
        return {"status": "error", "message": f"Internal error processing note: {e}"}


# --- Имитация внешних API (Sync) ---

def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Получает текущую погоду (имитация). (Sync)"""
    logger.info(f"--- Sync Function Call: get_current_weather(location='{location}', unit='{unit}') ---")
    time.sleep(0.1)
    location_lower = location.lower()
    if "tokyo" in location_lower: return {"location": location, "temperature": "15", "unit": unit, "description": "Облачно"}
    elif "san francisco" in location_lower: return {"location": location, "temperature": "20", "unit": unit, "description": "Солнечно"}
    elif "paris" in location_lower: return {"location": location, "temperature": "12", "unit": unit, "description": "Дождь"}
    else: return {"location": location, "temperature": "unknown", "unit": unit, "description": "Нет данных для этого местоположения"}

def get_stock_price(ticker_symbol: str) -> dict:
    """Получает цену акции (имитация). (Sync)"""
    logger.info(f"--- Sync Function Call: get_stock_price(ticker_symbol='{ticker_symbol}') ---")
    time.sleep(0.1)
    symbol = ticker_symbol.upper()
    if symbol == "GOOGL": return {"ticker": symbol, "price": "175.50", "currency": "USD", "change": "+1.20"}
    elif symbol == "AAPL": return {"ticker": symbol, "price": "190.20", "currency": "USD", "change": "-0.55"}
    else: return {"ticker": symbol, "price": "unknown", "currency": "USD", "change": "N/A"}

# --- Файловые операции (Async, с chat_id) ---

# --- ИСПРАВЛЕНИЕ: Добавляем недостающую функцию read_file_from_env ---
async def read_file_from_env(chat_id: int, filename: str) -> str:
    """
    Читает файл из ИЗОЛИРОВАННОЙ директории чата. (Async)
    Возвращает содержимое файла или строку с ошибкой.
    """
    logger.info(f"--- Async Function Call: read_file_from_env(chat_id={chat_id}, filename='{filename}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)
    if not is_safe or not filepath:
        return "Error: Access denied or invalid filename."
    try:
        # Используем aiofiles.os для асинхронных проверок
        if not await aiofiles.os.path.exists(filepath):
             logger.warning(f"File not found: {filepath}")
             return f"Error: File '{filename}' not found in the environment for this chat."
        if await aiofiles.os.path.isdir(filepath):
             logger.warning(f"Attempted to read directory as file: {filepath}")
             return f"Error: '{filename}' is a directory, not a file."

        async with aiofiles.open(filepath, mode="r", encoding='utf-8', errors='ignore') as f:
            content = await f.read()
        logger.debug(f"Successfully read file '{filename}' for chat {chat_id}, size={len(content)}")

        MAX_READ_SIZE = 100 * 1024 # 100 KB
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > MAX_READ_SIZE:
             logger.warning(f"File '{filename}' chat {chat_id} too large ({len(content_bytes)} bytes). Truncating.")
             truncated_content = content_bytes[:MAX_READ_SIZE].decode('utf-8', errors='ignore')
             return truncated_content + "\n... [File truncated due to size limit]"
        return content
    except Exception as e:
        logger.error(f"Error reading file '{filename}' async for chat {chat_id}: {e}", exc_info=True)
        return f"Error reading file '{filename}': {e}"

async def write_file_to_env(chat_id: int, filename: str, content: str) -> str:
    """
    Записывает текст в файл в ИЗОЛИРОВАННОЙ директории чата. (Async)
    Возвращает строку со статусом операции.
    """
    logger.info(f"--- Async Function Call: write_file_to_env(chat_id={chat_id}, filename='{filename}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)
    if not is_safe or not filepath:
        return "Error: Access denied or invalid filename."
    try:
        _ensure_chat_dir_exists(chat_id)
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
             os.makedirs(parent_dir, exist_ok=True)
             logger.info(f"Created parent directory: {parent_dir}")

        MAX_WRITE_SIZE = 500 * 1024 # 500 KB
        content_bytes = content.encode('utf-8')
        if len(content_bytes) > MAX_WRITE_SIZE:
             logger.error(f"Content for '{filename}' chat {chat_id} too large ({len(content_bytes)} bytes). Write aborted.")
             return f"Error: Content too large to write (max {MAX_WRITE_SIZE // 1024} KB)."

        async with aiofiles.open(filepath, mode="w", encoding='utf-8') as f:
            await f.write(content)
        logger.info(f"Successfully wrote file '{filename}' for chat {chat_id}, size={len(content_bytes)}")
        return f"File '{filename}' written successfully."
    except Exception as e:
        logger.error(f"Error writing file '{filename}' async for chat {chat_id}: {e}", exc_info=True)
        return f"Error writing file '{filename}': {e}"


async def create_file_in_env(chat_id: int, filename: str) -> str:
     """
     Создает новый пустой файл в ИЗОЛИРОВАННОЙ директории чата. (Async)
     Возвращает строку со статусом операции.
     """
     logger.info(f"--- Async Function Call: create_file_in_env(chat_id={chat_id}, filename='{filename}') ---")
     is_safe, filepath = _get_safe_chat_path(chat_id, filename)
     if not is_safe or not filepath:
          return "Error: Access denied or invalid filename."
     try:
          # Используем aiofiles.os
          if await aiofiles.os.path.exists(filepath):
               is_dir = await aiofiles.os.path.isdir(filepath)
               entity_type = "directory" if is_dir else "file"
               logger.warning(f"Cannot create: {entity_type} '{filename}' already exists at {filepath}")
               return f"Error: {entity_type.capitalize()} '{filename}' already exists for this chat."

          _ensure_chat_dir_exists(chat_id)
          parent_dir = os.path.dirname(filepath)
          if parent_dir and not os.path.exists(parent_dir):
               os.makedirs(parent_dir, exist_ok=True)
               logger.info(f"Created parent directory: {parent_dir}")

          async with aiofiles.open(filepath, mode="w", encoding='utf-8') as f:
               await f.write("")
          logger.info(f"Successfully created empty file '{filename}' for chat {chat_id}")
          return f"File '{filename}' created successfully."
     except Exception as e:
          logger.error(f"Error creating file '{filename}' async for chat {chat_id}: {e}", exc_info=True)
          return f"Error creating file '{filename}': {e}"

# --- Выполнение команд и скриптов (Sync, с chat_id) ---

def execute_python_script_in_env(chat_id: int, filename: str) -> str:
    """
    Выполняет Python-скрипт из ИЗОЛИРОВАННОЙ директории чата. (Sync)
    Возвращает строку с выводом скрипта или ошибкой.
    """
    logger.info(f"--- Sync Function Call: execute_python_script_in_env(chat_id={chat_id}, filename='{filename}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)
    chat_dir = os.path.join(ABS_ENV_DIR, str(chat_id))
    if not is_safe or not filepath: return "Error: Access denied or invalid filename."
    try: _ensure_chat_dir_exists(chat_id)
    except IOError as e: return f"Error: Could not ensure chat directory exists: {e}"
    if not os.path.exists(filepath): return f"Error: Script '{filename}' not found in environment for chat {chat_id}."
    if not filename.lower().endswith(".py"): return "Error: Only Python scripts (.py) can be executed."
    if os.path.isdir(filepath): return f"Error: '{filename}' is a directory, not a script."

    try:
        relative_filename = os.path.basename(filepath)
        process = subprocess.run(
            ["python", relative_filename], capture_output=True, text=True, timeout=30, cwd=chat_dir, check=False,
            encoding='utf-8', errors='ignore'
        )
        MAX_OUTPUT_LEN = 5000
        stdout = process.stdout[:MAX_OUTPUT_LEN] + ("...[truncated]" if len(process.stdout) > MAX_OUTPUT_LEN else "")
        stderr = process.stderr[:MAX_OUTPUT_LEN] + ("...[truncated]" if len(process.stderr) > MAX_OUTPUT_LEN else "")
        output = f"Exit Code: {process.returncode}\n\n"
        if stdout: output += f"--- STDOUT ---\n{stdout}\n\n"
        if stderr: output += f"--- STDERR ---\n{stderr}\n"
        logger.info(f"Executed script '{filename}' for chat {chat_id}, exit_code={process.returncode}")
        return output.strip()
    except subprocess.TimeoutExpired:
        logger.warning(f"Script '{filename}' timed out chat {chat_id}.")
        return f"Error: Script '{filename}' execution timed out (30 seconds)."
    except FileNotFoundError:
         logger.error(f"'python' command not found chat {chat_id}")
         return "Error: Python interpreter not found."
    except Exception as e:
        logger.error(f"Error executing script '{filename}' chat {chat_id}: {e}", exc_info=True)
        return f"Error executing script '{filename}': {e}"

def execute_terminal_command_in_env(chat_id: int, command: str) -> dict:
    """
    Выполняет команду в терминале ИЗОЛИРОВАННОЙ директории чата. (Sync)
    Возвращает словарь с stdout, stderr, returncode.
    """
    logger.info(f"--- Sync Function Call: execute_terminal_command_in_env(chat_id={chat_id}, command='{command}') ---")
    chat_dir = os.path.join(ABS_ENV_DIR, str(chat_id))
    try: _ensure_chat_dir_exists(chat_id)
    except IOError as e: return {"stdout": "", "stderr": f"Error: Could not ensure chat directory exists: {e}", "returncode": -99}

    forbidden_commands = ['sudo', 'rm -rf', 'mv /', 'mkfs', ':', 'shutdown', 'reboot']
    forbidden_patterns = ['cd ..', 'cd /']
    command_to_check = command.strip(); command_lower = command_to_check.lower(); command_parts = command_to_check.split()
    is_forbidden = False
    if not command_parts: is_forbidden = True
    else:
        if command_parts[0].lower() in forbidden_commands: is_forbidden = True
        for pattern in forbidden_patterns:
            if pattern in command_lower: is_forbidden = True; break
        if command_parts[0] == 'rm' and len(command_parts) > 1 and command_parts[1] != '-i': logger.warning(f"Risky 'rm' command chat {chat_id}: {command}") # is_forbidden = True

    if is_forbidden:
         logger.error(f"Forbidden terminal command chat {chat_id}: {command}")
         return {"stdout": "", "stderr": "Error: Command execution forbidden by security policy.", "returncode": -1}
    # if '|' in command or '>' in command or '<' in command: # Запрет пайпов/редиректов
    #       logger.warning(f"Risky command with pipe/redirect chat {chat_id}: {command}")
    #       return {"stdout": "", "stderr": "Error: Pipes and redirections are disabled.", "returncode": -1}

    try:
        process = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60,
            cwd=chat_dir, check=False, encoding='utf-8', errors='ignore'
        )
        MAX_OUTPUT_LEN = 5000
        stdout = process.stdout[:MAX_OUTPUT_LEN] + ("...[truncated]" if len(process.stdout) > MAX_OUTPUT_LEN else "")
        stderr = process.stderr[:MAX_OUTPUT_LEN] + ("...[truncated]" if len(process.stderr) > MAX_OUTPUT_LEN else "")
        logger.info(f"Executed command '{command}' chat {chat_id}, exit_code={process.returncode}")
        return {"stdout": stdout, "stderr": stderr, "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        logger.warning(f"Command '{command}' timed out chat {chat_id}.")
        return {"stdout": "", "stderr": "Error: Command execution timed out (60 seconds).", "returncode": -2}
    except Exception as e:
        logger.error(f"Error executing command '{command}' chat {chat_id}: {e}", exc_info=True)
        return {"stdout": "", "stderr": f"Error executing command: {e}", "returncode": -3}


# --- Редактирование файлов (Async/Sync, с chat_id) ---

async def edit_file_content(chat_id: int, filename: str, search_string: str, replace_string: str) -> str:
    """
    Редактирует файл в ИЗОЛИРОВАННОЙ директории чата, заменяя текст. (Async)
    Возвращает строку со статусом операции.
    """
    logger.info(f"--- Async Function Call: edit_file_content(chat_id={chat_id}, filename='{filename}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)
    if not is_safe or not filepath: return "Error: Access denied or invalid filename."
    try:
        # Используем aiofiles.os
        if not await aiofiles.os.path.exists(filepath): return f"Error: File '{filename}' not found chat {chat_id}."
        if await aiofiles.os.path.isdir(filepath): return f"Error: '{filename}' is a directory."

        async with aiofiles.open(filepath, "r+", encoding='utf-8', errors='ignore') as f:
            original_content = await f.read()
            if search_string not in original_content:
                logger.info(f"Search string not found '{filename}' chat {chat_id}.")
                return f"Info: Search string was not found in '{filename}'. No changes made."
            modified_content = original_content.replace(search_string, replace_string)
            await f.seek(0); await f.write(modified_content); await f.truncate()
        logger.info(f"Edited file '{filename}' chat {chat_id}.")
        return f"File '{filename}' edited successfully. Replaced occurrences of search string."
    except Exception as e:
        logger.error(f"Error editing file '{filename}' async chat {chat_id}: {e}", exc_info=True)
        return f"Error editing file '{filename}': {e}"


# --- AST и JSON редактирование (Sync, с chat_id) ---

class ReplaceCodeTransformer(ast.NodeTransformer):
    """AST Transformer для замены узла (функции или класса)."""
    def __init__(self, block_type: str, block_name: str, new_code_node: ast.AST):
        self.block_type = block_type; self.block_name = block_name
        self.new_code_node = new_code_node; self.replaced = False
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if self.block_type == "function" and node.name == self.block_name:
            if isinstance(self.new_code_node, ast.FunctionDef):
                logger.info(f"Replacing function '{self.block_name}'.")
                if node.decorator_list and not self.new_code_node.decorator_list: self.new_code_node.decorator_list = node.decorator_list
                self.replaced = True; return self.new_code_node
            else: logger.error(f"AST Type Mismatch: Expected FunctionDef for '{self.block_name}', got {type(self.new_code_node)}."); return node
        return self.generic_visit(node)
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        if self.block_type == "class" and node.name == self.block_name:
            if isinstance(self.new_code_node, ast.ClassDef):
                logger.info(f"Replacing class '{self.block_name}'.")
                if node.decorator_list and not self.new_code_node.decorator_list: self.new_code_node.decorator_list = node.decorator_list
                self.replaced = True; return self.new_code_node
            else: logger.error(f"AST Type Mismatch: Expected ClassDef for '{self.block_name}', got {type(self.new_code_node)}."); return node
        return self.generic_visit(node)

def replace_code_block_ast(chat_id: int, filename: str, block_type: str, block_name: str, new_code_block: str) -> str:
    """
    Заменяет блок кода (функцию или класс) в Python-файле ИЗОЛИРОВАННОЙ директории чата. (Sync)
    Возвращает строку со статусом операции.
    """
    logger.info(f"--- Sync Function Call: replace_code_block_ast(chat_id={chat_id}, file='{filename}', type='{block_type}', name='{block_name}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)
    if not is_safe or not filepath: return "Error: Access denied or invalid filename."
    try: _ensure_chat_dir_exists(chat_id)
    except IOError as e: return f"Error: Could not ensure chat directory exists: {e}"
    if not os.path.exists(filepath): return f"Error: File '{filename}' not found chat {chat_id}."
    if not filename.lower().endswith(".py"): return "Error: Can only edit Python (.py) files."
    if os.path.isdir(filepath): return f"Error: '{filename}' is a directory."
    if block_type not in ["function", "class"]: return f"Error: Invalid block_type '{block_type}'."

    try:
        with open(filepath, "r", encoding='utf-8') as f: source_code = f.read()
        tree = ast.parse(source_code, filename=filename)
        try: new_code_tree = ast.parse(new_code_block)
        except SyntaxError as se: logger.error(f"Syntax error parsing new_code_block for {block_type} '{block_name}': {se}"); return f"Error: Syntax error in new code: {se}"
        except Exception as parse_e: logger.error(f"Error parsing new_code_block {block_type} '{block_name}': {parse_e}", exc_info=True); return f"Error: Could not parse new code: {parse_e}"

        if not new_code_tree.body or len(new_code_tree.body) != 1: return "Error: 'new_code_block' must contain exactly one function/class def."
        new_node = new_code_tree.body[0]
        if block_type == "function" and not isinstance(new_node, ast.FunctionDef): return f"Error: block_type=function but code is {type(new_node)}."
        if block_type == "class" and not isinstance(new_node, ast.ClassDef): return f"Error: block_type=class but code is {type(new_node)}."

        transformer = ReplaceCodeTransformer(block_type, block_name, new_node)
        new_tree = transformer.visit(tree)
        if not transformer.replaced: return f"Error: {block_type.capitalize()} '{block_name}' not found in '{filename}'."

        new_source_code = ast.unparse(new_tree)
        with open(filepath, "w", encoding='utf-8') as f: f.write(new_source_code)
        logger.info(f"Replaced {block_type} '{block_name}' in '{filename}' chat {chat_id}.")
        return f"Successfully replaced {block_type} '{block_name}' in '{filename}'."
    except ast.ASTError as e: logger.error(f"AST Error parsing original file '{filename}' chat {chat_id}: {e}", exc_info=True); return f"Error parsing Python code in '{filename}': {e}"
    except Exception as e: logger.error(f"Error editing Python file '{filename}' AST chat {chat_id}: {e}", exc_info=True); return f"Error editing Python file via AST: {e}"


def edit_json_file(chat_id: int, filename: str, json_path: str, new_value_json: str) -> str:
    """
    Редактирует JSON-файл в ИЗОЛИРОВАННОЙ директории чата по указанному пути. (Sync)
    Возвращает строку со статусом операции.
    """
    logger.info(f"--- Sync Function Call: edit_json_file(chat_id={chat_id}, file='{filename}', path='{json_path}') ---")
    is_safe, filepath = _get_safe_chat_path(chat_id, filename)

    # Проверки безопасности и существования файла
    if not is_safe or not filepath: return "Error: Access denied or invalid filename."
    try: _ensure_chat_dir_exists(chat_id)
    except IOError as e: return f"Error: Could not ensure chat directory exists: {e}"
    if not os.path.exists(filepath): return f"Error: File '{filename}' not found for chat {chat_id}."
    if not filename.lower().endswith(".json"): logger.warning(f"File '{filename}' might not be a JSON file.")
    if os.path.isdir(filepath): return f"Error: '{filename}' is a directory."

    try:
        # Чтение JSON
        with open(filepath, "r", encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error reading file '{filename}' for chat {chat_id}: {e}")
                return f"Error decoding JSON from file '{filename}': {e}"

        # Парсинг нового значения
        try:
            new_value = json.loads(new_value_json)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse new_value_json '{new_value_json}' as JSON for chat {chat_id}. Using as raw string.")
            new_value = new_value_json

        # --- Логика парсинга json_path и установки значения ---
        keys: List[Union[str, int]] = []
        current_part = ""
        in_bracket = False
        in_quotes = False
        quote_char = ''

        path_iterator = iter(enumerate(json_path))
        for i, char in path_iterator:
            # Обработка кавычек для ключей словаря
            if char in ('"', "'") and not in_bracket:
                 if not in_quotes: # Открывающая кавычка
                     in_quotes = True
                     quote_char = char
                     # Если перед кавычкой был ключ без кавычек, добавляем его
                     if current_part:
                         keys.append(current_part)
                         current_part = ""
                 elif char == quote_char: # Закрывающая кавычка
                      in_quotes = False
                      keys.append(current_part) # Добавляем ключ в кавычках
                      current_part = ""
                      # --- ИСПРАВЛЕНИЕ SyntaxError ---
                      # Пропускаем точку после кавычки, если она есть
                      try:
                          next_char_index = i + 1
                          if next_char_index < len(json_path) and json_path[next_char_index] == '.':
                              next(path_iterator, None) # Продвигаем итератор, чтобы пропустить точку
                      except IndexError: # Дошли до конца строки
                          pass
                 else: # Символ внутри кавычек
                      current_part += char
                 continue # Переходим к следующей итерации после обработки кавычки

            # Если мы внутри кавычек, просто собираем ключ
            if in_quotes:
                current_part += char
                continue

            # Обработка символов вне кавычек
            if char == '.' and not in_bracket: # Разделитель ключей
                if not current_part and not keys: return f"Error: JSON path cannot start with '.'";
                if not current_part: return f"Error: Invalid JSON path near index {i} (empty key)."
                keys.append(current_part); current_part = ""
            elif char == '[' and not in_bracket: # Начало индекса списка
                if current_part: keys.append(current_part); current_part = "";
                in_bracket = True
            elif char == ']' and in_bracket: # Конец индекса списка
                try: keys.append(int(current_part)) # Индекс списка
                except ValueError: return f"Error: Invalid list index '{current_part}' in JSON path."
                current_part = ""; in_bracket = False
                # --- ИСПРАВЛЕНИЕ SyntaxError ---
                # Пропускаем точку после скобки, если она есть
                try:
                    next_char_index = i + 1
                    if next_char_index < len(json_path) and json_path[next_char_index] == '.':
                         next(path_iterator, None) # Продвигаем итератор
                except IndexError:
                    pass
            elif in_bracket: # Внутри скобок собираем индекс
                 if not char.isdigit(): return f"Error: Non-digit character '{char}' inside list index brackets."
                 current_part += char
            else: # Собираем ключ словаря (без кавычек)
                current_part += char

        # Проверки после цикла
        if in_quotes: return f"Error: Unmatched '{quote_char}' in JSON path."
        if in_bracket: return "Error: Unmatched '[' in JSON path."
        if current_part: keys.append(current_part) # Добавляем последнюю часть пути
        if not keys: return "Error: Empty or invalid JSON path provided."
        # --- Конец разбора пути ---

        # --- Логика навигации и установки значения (остается как в прошлом исправлении) ---
        temp_data: Any = data; parent: Optional[Union[Dict, List]] = None; last_key_in_parent: Optional[Union[str, int]] = None
        for i, key in enumerate(keys[:-1]):
            parent = temp_data; last_key_in_parent = key
            try:
                if isinstance(temp_data, list):
                    if not isinstance(key, int): return f"Error: Non-integer key '{key}' for list at path index {i}."
                    temp_data = temp_data[key]
                elif isinstance(temp_data, dict):
                    key_str = str(key) if not isinstance(key, str) else key
                    if key_str not in temp_data: return f"Error: Key '{key_str}' not found at path index {i}."
                    temp_data = temp_data[key_str]
                else: return f"Error: Cannot access key '{key}' on element type {type(temp_data)} at path index {i}."
            except IndexError: return f"Error: Index {key} out of bounds at path index {i}."
            except KeyError: return f"Error: Key '{key}' not found at path index {i}."
            except Exception as nav_e: return f"Error navigating path at index {i} ('{key}'): {nav_e}"

        last_key = keys[-1]
        try:
            if isinstance(temp_data, list):
                if not isinstance(last_key, int): return f"Error: Final key '{last_key}' must be integer for list."
                if not (0 <= last_key < len(temp_data)): return f"Error: Final index {last_key} out of bounds (size {len(temp_data)})."
                temp_data[last_key] = new_value
            elif isinstance(temp_data, dict):
                 last_key_str = str(last_key) if not isinstance(last_key, str) else last_key
                 temp_data[last_key_str] = new_value
            elif parent is not None and last_key_in_parent is not None:
                 if isinstance(parent, list) and isinstance(last_key_in_parent, int): parent[last_key_in_parent] = new_value; logger.warning(f"Overwrote primitive at list index {last_key_in_parent}")
                 elif isinstance(parent, dict): parent[str(last_key_in_parent)] = new_value; logger.warning(f"Overwrote primitive for key '{last_key_in_parent}'")
                 else: return f"Error: Cannot set value. Invalid parent type {type(parent)} or key/index type {type(last_key_in_parent)}."
            elif len(keys) == 1: data = new_value; logger.warning(f"Overwrote entire JSON file '{filename}' chat {chat_id}.")
            else: return f"Error: Cannot set value at '{last_key}'. Parent is not list/dict or path invalid."
        except IndexError: return f"Error: Final index {last_key} out of bounds."
        except (ValueError, TypeError) as set_val_err: return f"Error setting final value: {set_val_err}"
        except Exception as set_e: return f"Error setting value at '{last_key}': {set_e}"
        # --- Конец установки значения ---

        # Запись обратно в файл
        with open(filepath, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info(f"Successfully edited JSON file '{filename}' at path '{json_path}' for chat {chat_id}.")
        return f"JSON file '{filename}' edited successfully at path '{json_path}'."

    # Блок except для ошибок на уровне файла (не JSON)
    except Exception as e:
        logger.error(f"Error processing JSON file '{filename}' for chat {chat_id}: {e}", exc_info=True)
        return f"Error processing JSON file '{filename}': {e}"