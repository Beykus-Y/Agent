import time
import logging
import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Tuple

# --- Aiogram Imports ---
from aiogram import types
from aiogram.enums import ChatType

# --- Локальные импорты ---
import database
from bot_loader import bot
from bot_utils import gemini_history_to_dict_list, escape_markdown_v2
from bot_processing import process_gemini_fc_cycle # Импортируем уже измененную функцию
import gemini_api

# --- Типы Google ---
try:
    from google.ai import generativelanguage as glm
    Content = glm.Content
    Part = glm.Part
    logger = logging.getLogger(__name__)
    logger.debug("Successfully imported Content and Part from google.ai.generativelanguage (as glm)")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.critical(f"Failed to import google types (glm) in handlers.common: {e}", exc_info=True)
    Content = None # type: ignore
    Part = None # type: ignore

logger = logging.getLogger(__name__)

# Функция prepare_gemini_session_history остается без изменений

async def prepare_gemini_session_history(
    chat_id: int,
    user_id: int, # ID ТЕКУЩЕГО пользователя (для заметок)
    add_notes: bool = True
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Получает историю из БД (которая теперь включает user_id), заметки,
    форматирует историю для модели (НЕ добавляя префиксы User ID),
    и проверяет последний элемент на незавершенный FC.

    Возвращает:
        - history_to_pass_to_gemini: Отформатированный список словарей для model.start_chat().
        - original_history_len_for_save: Длина отформатированной истории ДО очистки FC.
    """
    logger.debug(f"Preparing history for chat={chat_id}, current_user={user_id}")
    # get_chat_history возвращает user_id в записях 'user'
    history_from_db = await database.get_chat_history(chat_id)
    user_notes = {}
    if add_notes:
        user_notes = await database.get_user_notes(user_id)

    # Список для истории, которая будет передана модели
    formatted_history_list_dict: List[Dict[str, Any]] = []

    # 1. Добавляем заметки о ТЕКУЩЕМ пользователе
    if user_notes:
        notes_list = [f"- {escape_markdown_v2(cat)}: {escape_markdown_v2(val)}" for cat, val in user_notes.items()]
        safe_user_id_str = escape_markdown_v2(str(user_id))
        notes_context = f"\\_\\_\\_System Note\\_\\_\\_\n*Known info about user {safe_user_id_str}*:*\n" + "\n".join(notes_list)
        formatted_history_list_dict.append({"role": "model", "parts": [{"text": notes_context}]})
        logger.info(f"Added {len(user_notes)} notes to context for current user {user_id} in chat {chat_id}.")

    # 2. Добавляем историю из БД КАК ЕСТЬ (без префиксов)
    for entry in history_from_db:
        role = entry.get("role")
        parts = entry.get("parts")
        # db_user_id = entry.get("user_id") # Мы больше не используем его для форматирования

        if not role or not parts:
            logger.warning(f"Skipping history entry with missing role or parts: {entry}")
            continue

        # Просто добавляем запись как есть (parts уже должны быть list of dicts)
        # Роль 'user' будет просто содержать текст сообщения пользователя
        # Роль 'model' будет содержать текст ответа модели
        formatted_history_list_dict.append({"role": role, "parts": parts})

    # 3. Запоминаем длину отформатированной истории ДО очистки
    original_history_len_for_save = len(formatted_history_list_dict)

    # 4. Проверка и очистка ПОСЛЕДНЕГО сообщения модели от незавершенного FC (как раньше)
    history_to_pass_to_gemini = formatted_history_list_dict # По умолчанию
    if formatted_history_list_dict:
        last_entry_dict = formatted_history_list_dict[-1]
        if last_entry_dict.get("role") == 'model':
            parts_list_dict = last_entry_dict.get("parts", [])
            has_fc = any('function_call' in part_dict for part_dict in parts_list_dict)
            if has_fc:
                logger.warning(f"Last history entry (model) contains function_call. Starting session WITHOUT this entry for chat {chat_id}.")
                history_to_pass_to_gemini = formatted_history_list_dict[:-1]
                original_history_len_for_save = len(history_to_pass_to_gemini)

    logger.debug(f"Prepared history len={len(history_to_pass_to_gemini)} for chat={chat_id}")
    return history_to_pass_to_gemini, original_history_len_for_save


# --- ИЗМЕНЕНО: Функция run_gemini_interaction теперь возвращает имя последней функции ---
async def run_gemini_interaction(
    model_instance: Any,
    initial_history: List[Dict[str, Any]],
    user_input: str,
    available_functions: Dict[str, Callable],
    max_steps: int,
    chat_id: int,
    user_id: int
) -> Tuple[Optional[List[Content]], Optional[str], Optional[str]]: # <--- ИЗМЕНЕН ТИП: Возвращает историю, ошибку, имя функции
    """
    Запускает сессию Gemini, отправляет сообщение и обрабатывает FC.
    Возвращает финальную историю, сообщение об ошибке (если есть) и имя последней успешно вызванной функции.
    """
    logger.info(f"Running Gemini interaction for chat={chat_id}, current_user={user_id}")
    final_history_obj_list: Optional[List[Content]] = None
    # --- ИЗМЕНЕНО: Инициализируем переменную для имени функции ---
    last_called_func_name: Optional[str] = None
    # ------------------------------------------------------
    error_message: Optional[str] = None

    try:
        try:
             history_json_for_log = json.dumps(initial_history, indent=2, ensure_ascii=False, default=str)
             logger.debug(f"Formatted history passed to model.start_chat() for chat {chat_id}:\n{history_json_for_log}")
        except Exception as log_e:
             logger.error(f"Failed to serialize formatted history for logging chat {chat_id}: {log_e}")
             logger.info(f"Formatted history passed to model.start_chat() (raw preview) for chat {chat_id}: {str(initial_history)[:1000]}...")

        chat_session = model_instance.start_chat(history=initial_history)

        loop = asyncio.get_running_loop()
        current_response = await loop.run_in_executor(
            None, gemini_api.send_message_to_gemini, model_instance, chat_session, user_input
        )

        if current_response is None:
             error_message = "Failed to communicate with the AI model (initial request)."
             logger.error(f"{error_message} Chat: {chat_id}")
             # --- ИЗМЕНЕНО: Возвращаем None для имени функции ---
             return None, error_message, None
             # -----------------------------------------------

        if Content is None:
            error_message = "Internal configuration error: AI type system failed."
            logger.critical(error_message)
            # --- ИЗМЕНЕНО: Возвращаем None для имени функции ---
            return None, error_message, None
            # -----------------------------------------------

        # --- ИЗМЕНЕНО: Принимаем имя последней функции из process_gemini_fc_cycle ---
        final_history_obj_list, last_called_func_name = await process_gemini_fc_cycle(
            model_instance=model_instance, chat_session=chat_session,
            available_functions_map=available_functions,
            max_steps=max_steps,
            original_chat_id=chat_id,
            original_user_id=user_id
        )
        # -------------------------------------------------------------------------

        if final_history_obj_list is None:
             error_message = "AI model processing cycle failed critically."
             logger.error(f"{error_message} Chat: {chat_id}")
             # last_called_func_name уже будет None из-за возврата process_gemini_fc_cycle
             return None, error_message, last_called_func_name
        elif not final_history_obj_list:
             error_message = "AI model processing cycle resulted in empty history."
             logger.error(f"{error_message} Chat: {chat_id}")
             # Возвращаем имя, если оно было на последнем шаге (хотя история пуста)
             return None, error_message, last_called_func_name

    except ValueError as ve:
        error_message = f"Failed to start AI session: {ve}"
        logger.error(f"ValueError during Gemini session start chat {chat_id}: {ve}", exc_info=True)
        # last_called_func_name будет None
    except Exception as e:
        error_message = f"An unexpected error occurred during AI interaction: {e}"
        logger.error(f"Unexpected error in run_gemini_interaction for chat {chat_id}: {e}", exc_info=True)
        # last_called_func_name будет None

    # --- ИЗМЕНЕНО: Возвращаем имя последней функции ---
    return final_history_obj_list, error_message, last_called_func_name
    # -----------------------------------------------

# Функции extract_final_text, send_final_response, save_new_history_entries остаются без изменений

def extract_final_text(final_history_obj_list: Optional[List[Content]]) -> Optional[str]:
    # ... (код без изменений) ...
    if Content is None:
        logger.error("Cannot extract text: Google type 'Content' not imported.")
        return "[Error: Internal type configuration issue]"
    if not final_history_obj_list:
        logger.warning("Cannot extract text from empty history list.")
        return None
    try:
        last_model_content = final_history_obj_list[-1]
        if not isinstance(last_model_content, Content):
             logger.error(f"Last history entry is not a Content object: {type(last_model_content)}")
             return "[Error: Invalid history structure]"
        if last_model_content.role == 'model':
            text_parts = [part.text for part in last_model_content.parts if hasattr(part, 'text')]
            if text_parts:
                final_response_text = "\n".join(text_parts).strip()
                logger.debug(f"Extracted final text (len={len(final_response_text)}).")
                return final_response_text
            else:
                logger.info("Final model response has no text part.")
                return None
        else:
            logger.warning("Last history entry not from model.")
            return None
    except (IndexError, AttributeError, TypeError) as e:
        logger.error(f"Error extracting final text: {e}", exc_info=True)
        return None

async def send_final_response(message: types.Message, final_response_text: Optional[str]):
    # ... (код без изменений) ...
    if not final_response_text:
        logger.info(f"No final text to send for message {message.message_id} in chat {message.chat.id}.")
        return
    chat_id = message.chat.id
    logger.info(f"Sending final text response to chat {chat_id}")
    escaped_final_text = escape_markdown_v2(final_response_text)
    try:
        await message.reply(escaped_final_text)
    except Exception as send_err:
        logger.error(f"Failed final reply to chat {chat_id}: {send_err}", exc_info=True)
        try:
            logger.info(f"Falling back to send_message for chat {chat_id}")
            await bot.send_message(chat_id, escaped_final_text)
        except Exception as send_err2:
            logger.error(f"Failed final message (no reply) to chat {chat_id}: {send_err2}", exc_info=True)

async def save_new_history_entries(
    chat_id: int,
    final_history_obj_list: Optional[List[Content]],
    original_history_len_for_save: int,
    current_user_id: int
):
    # ... (код без изменений) ...
    if Content is None:
        logger.error("Cannot save history: Google type 'Content' not imported.")
        return
    if not final_history_obj_list:
        logger.warning(f"Final history obj list empty/None, nothing save chat {chat_id}.")
        return
    if not all(isinstance(item, Content) for item in final_history_obj_list):
         logger.error(f"Cannot save history: Not all items in final_history_obj_list are Content objects.")
         return

    try:
        final_history_dict_list = gemini_history_to_dict_list(final_history_obj_list)

        new_history_entries = final_history_dict_list[original_history_len_for_save:]
        if not new_history_entries:
             logger.info(f"No new history entries to save for chat {chat_id}.")
             return

        logger.info(f"Attempting save {len(new_history_entries)} new history entries chat {chat_id}.")
        saved_count = 0
        for entry_dict in new_history_entries:
            role = entry_dict.get("role")
            parts_list_dict = entry_dict.get("parts", [])
            if not role or not parts_list_dict:
                logger.warning(f"Skipping invalid entry during history save (no role/parts): {entry_dict}")
                continue

            parts_to_save_dict = []
            user_id_to_save: Optional[int] = None

            if role == 'user':
                filtered_parts = []
                for p in parts_list_dict:
                     if isinstance(p, dict) and 'text' in p:
                          filtered_parts.append(p)
                parts_to_save_dict = filtered_parts
                user_id_to_save = current_user_id
            elif role == 'model':
                parts_to_save_dict = [p for p in parts_list_dict if isinstance(p, dict) and 'text' in p]
                user_id_to_save = None
            elif role == 'function':
                 logger.debug(f"Skipping history entry with role 'function' (FunctionResponse): {entry_dict}")
                 continue
            else:
                logger.warning(f"Skipping history entry with role '{role}' (not 'user' or 'model'): {entry_dict}")
                continue

            if parts_to_save_dict:
                 try:
                      await database.add_message_to_history(chat_id, role, parts_to_save_dict, user_id=user_id_to_save)
                      saved_count += 1
                 except Exception as db_save_err:
                      logger.error(f"Failed save history entry (role: {role}) chat {chat_id}: {db_save_err}", exc_info=True)
            else:
                 logger.debug(f"No valid parts left to save for role '{role}' after filtering, chat {chat_id}.")

        logger.info(f"Saved {saved_count} out of {len(new_history_entries)} possible new entries for chat {chat_id}.")

    except Exception as e:
        logger.error(f"Unexpected error during history saving for chat {chat_id}: {e}", exc_info=True)