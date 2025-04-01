# bot_handlers.py
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable

# --- Aiogram Imports ---
from aiogram import types, F
from aiogram.enums import ChatType
from aiogram.utils.markdown import hcode, hitalic

# --- Локальные импорты ---
import database
from bot_loader import dp, bot
from bot_utils import gemini_history_to_dict_list, escape_markdown_v2
from bot_processing import process_gemini_fc_cycle
import gemini_api

# --- Типы Google ---
try:
    from google.ai import generativelanguage as glm
    # --- ИСПРАВЛЕНИЕ: Content не нужен здесь, т.к. работаем со словарями ---
    Part = glm.Part
    # Content = glm.Content # Убираем Content
except ImportError as e:
    logging.critical(f"Failed to import google types in bot_handlers: {e}")
    Part = None
    # Content = None
    exit(1)

logger = logging.getLogger(__name__)


@dp.message(F.text & (F.chat.type == ChatType.PRIVATE))
async def handle_private_message(message: types.Message,
                                 pro_model: Any,
                                 available_pro_functions: Dict[str, Callable],
                                 max_pro_steps: int):
    """Обработчик для личных сообщений."""
    user = message.from_user
    chat = message.chat
    user_input = message.text or ""
    start_time = time.monotonic()

    if not user: logger.warning("Private message without user info?"); return
    logger.info(f"PRIVATE Msg from user={user.id} chat={chat.id}")

    try: await database.upsert_user_profile(user.id, user.username, user.first_name, user.last_name)
    except Exception as db_err: logger.error(f"Upsert profile failed user {user.id}: {db_err}", exc_info=True)

    if not pro_model:
        logger.critical("Pro model missing in workflow_data!")
        await message.reply(escape_markdown_v2("AI model is unavailable. Please try again later."))
        return

    logger.info(f"Running Pro model for private chat={chat.id}, user={user.id}")
    final_response_text = None
    original_history_len_for_save = 0 # Длина до добавления заметок
    try:
        # Получаем историю и заметки (история уже в виде списка словарей)
        history_from_db = await database.get_chat_history(chat.id)
        user_notes = await database.get_user_notes(user.id)

        # --- ИСПРАВЛЕНИЕ: Работаем только со словарями ---
        # initial_pro_history теперь List[Dict[str, Any]]
        initial_pro_history: List[Dict[str, Any]] = []

        # Добавляем заметки (как dict)
        if user_notes:
            notes_list = [f"- {escape_markdown_v2(cat)}: {escape_markdown_v2(val)}" for cat, val in user_notes.items()]
            safe_user_id_str = escape_markdown_v2(str(user.id))
            notes_context = f"\\_\\_\\_System Note\\_\\_\\_\n*Known info about user {safe_user_id_str}*:*\n" + "\n".join(notes_list)
            initial_pro_history.append({"role": "model", "parts": [{"text": notes_context}]})
            logger.info(f"Added {len(user_notes)} notes to Pro context private chat {chat.id}.")

        # Добавляем историю из БД (она уже list of dicts)
        initial_pro_history.extend(history_from_db)
        original_history_len_for_save = len(initial_pro_history) # Запоминаем длину для сохранения

        # --- ИСПРАВЛЕНИЕ: Проверка и очистка ИСТОРИИ СЛОВАРЕЙ ---
        history_to_pass_to_gemini = initial_pro_history # По умолчанию передаем всю историю
        if initial_pro_history:
            last_entry_dict = initial_pro_history[-1]
            if last_entry_dict.get("role") == 'model':
                # Проверяем наличие function_call в словарях частей
                parts_list_dict = last_entry_dict.get("parts", [])
                has_fc = any('function_call' in part_dict for part_dict in parts_list_dict)
                if has_fc:
                    logger.warning(f"Last history entry contains function_call. Starting chat session WITHOUT this entry for chat {chat.id}.")
                    # Передаем историю БЕЗ последнего элемента
                    history_to_pass_to_gemini = initial_pro_history[:-1]
                    # Обновляем длину для сохранения, т.к. последний элемент не будет в сессии
                    original_history_len_for_save = len(history_to_pass_to_gemini)

        # Создаем сессию чата, передавая список СЛОВАРЕЙ
        # Передаем очищенную историю
        pro_chat_session = pro_model.start_chat(history=history_to_pass_to_gemini)

        # Отправляем первое сообщение пользователя
        loop = asyncio.get_running_loop()
        current_response = await loop.run_in_executor(None, gemini_api.send_message_to_gemini, pro_model, pro_chat_session, user_input)
        # После send_message_to_gemini chat_session.history обновится (если успешно)
        if not current_response: raise ValueError("Initial response from Gemini was None.")

        # Запускаем цикл обработки FC
        # final_history_obj_list будет содержать Content объекты
        final_history_obj_list, _ = await process_gemini_fc_cycle(
            model_instance=pro_model, chat_session=pro_chat_session,
            available_functions_map=available_pro_functions,
            max_steps=max_pro_steps,
            original_chat_id=chat.id, original_user_id=user.id
        )

        # --- Извлечение финального текста ---
        if final_history_obj_list:
             # Работаем с Content объектами здесь
             last_model_content = final_history_obj_list[-1]
             if last_model_content.role == 'model':
                  text_parts = [part.text for part in last_model_content.parts if hasattr(part, 'text')]
                  if text_parts: final_response_text = "\n".join(text_parts).strip()
                  else: logger.info(f"Pro final response chat {chat.id} has no text part.")
             else: logger.warning(f"Last history entry chat {chat.id} not from model.")
        else: logger.error(f"Final history (obj list) empty/None chat {chat.id}.")

        # --- Отправка финального ответа ---
        if final_response_text:
             logger.info(f"Sending final text response private chat {chat.id}")
             escaped_final_text = escape_markdown_v2(final_response_text)
             try: await message.reply(escaped_final_text)
             except Exception as send_err:
                  logger.error(f"Failed final reply {chat.id}: {send_err}", exc_info=True)
                  try: await bot.send_message(chat.id, escaped_final_text)
                  except Exception as send_err2: logger.error(f"Failed final message (no reply) {chat.id}: {send_err2}", exc_info=True)
        elif not final_response_text and final_history_obj_list:
             logger.info(f"No final text send chat {chat.id}, processing finished.")

        # --- Сохранение истории ---
        if final_history_obj_list:
             # Конвертируем финальную историю (Content) в dict для БД
             final_history_dict_list = gemini_history_to_dict_list(final_history_obj_list)
             # Определяем новые записи, сравнивая с длиной ДО СТАРТА СЕССИИ
             new_history_entries = final_history_dict_list[original_history_len_for_save:] # Используем сохраненную длину
             logger.info(f"Attempting save {len(new_history_entries)} new history entries chat {chat.id}.")
             saved_count = 0
             # Итерируем по НОВЫМ записям (которые уже dict)
             for entry_dict in new_history_entries:
                 role = entry_dict.get("role")
                 parts_list_dict = entry_dict.get("parts", []) # Это УЖЕ список словарей
                 if not role or not parts_list_dict:
                     logger.warning(f"Skipping invalid entry during history save: {entry_dict}")
                     continue

                 # Отфильтровываем части с FC/FR для роли 'model'
                 parts_to_save_dict = parts_list_dict
                 if role == "model":
                      parts_to_save_dict = [p_dict for p_dict in parts_list_dict if 'function_call' not in p_dict and 'function_response' not in p_dict]

                 # Передаем отфильтрованный список СЛОВАРЕЙ в add_message_to_history
                 if parts_to_save_dict:
                      try:
                           # Передаем СПИСОК СЛОВАРЕЙ
                           await database.add_message_to_history(chat.id, role, parts_to_save_dict)
                           saved_count += 1
                      except Exception as db_save_err:
                           logger.error(f"Failed save history entry (role: {role}) chat {chat.id}: {db_save_err}", exc_info=True)
             logger.info(f"Saved {saved_count} new entries chat {chat.id}.")
        else:
             logger.warning(f"Final history obj list empty/None, nothing save chat {chat.id}.")

    except Exception as e:
        logger.error(f"Error processing private message chat {chat.id}: {e}", exc_info=True)
        try:
             error_msg_escaped = escape_markdown_v2(f"An internal error occurred: {e}")
             await message.reply(error_msg_escaped)
        except Exception as send_err: logger.error(f"Failed send error reply chat {chat.id}: {send_err}", exc_info=True)

    end_time = time.monotonic()
    logger.info(f"Finished processing PRIVATE msg {chat.id} in {end_time - start_time:.2f} sec.")


@dp.message(F.text & (F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP})))
async def handle_group_message(message: types.Message,
                               lite_model: Any,
                               pro_model: Any,
                               available_lite_functions: Dict[str, Callable],
                               available_pro_functions: Dict[str, Callable],
                               max_lite_steps: int,
                               max_pro_steps: int):
    """Обработчик для сообщений в группах."""
    # ... (код до запуска Pro модели) ...
    if run_pro_model:
        # ...
        original_history_len_for_save_group = 0 # Длина для сохранения
        try:
            # Получаем историю и заметки
            history_from_db_group = await database.get_chat_history(chat.id)
            user_notes_group = await database.get_user_notes(user.id)

            # Формируем начальную историю (List[Dict])
            initial_pro_history_group: List[Dict[str, Any]] = []
            if user_notes_group:
                 # ... (добавление заметок как dict) ...
                 notes_list_group = [f"- {escape_markdown_v2(cat)}: {escape_markdown_v2(val)}" for cat, val in user_notes_group.items()]
                 safe_user_id_str_group = escape_markdown_v2(str(user.id))
                 notes_context_group = f"\\_\\_\\_System Note\\_\\_\\_\n*Known info about user {safe_user_id_str_group}*:*\n" + "\n".join(notes_list_group)
                 initial_pro_history_group.append({"role": "model", "parts": [{"text": notes_context_group}]})
                 logger.info(f"Added {len(user_notes_group)} notes Pro context group {chat.id}.")

            initial_pro_history_group.extend(history_from_db_group)
            original_history_len_for_save_group = len(initial_pro_history_group)

            # --- ИСПРАВЛЕНИЕ: Очистка истории СЛОВАРЕЙ от незавершенного FC ---
            history_to_pass_to_gemini_group = initial_pro_history_group
            if initial_pro_history_group:
                last_entry_dict_group = initial_pro_history_group[-1]
                if last_entry_dict_group.get("role") == 'model':
                    parts_list_dict_group = last_entry_dict_group.get("parts", [])
                    has_fc_group = any('function_call' in part_dict for part_dict in parts_list_dict_group)
                    if has_fc_group:
                        logger.warning(f"Last history entry contains function_call. Starting chat session WITHOUT this entry for group chat {chat.id}.")
                        history_to_pass_to_gemini_group = initial_pro_history_group[:-1]
                        original_history_len_for_save_group = len(history_to_pass_to_gemini_group)

            # Создаем сессию Pro модели, передавая список СЛОВАРЕЙ
            pro_chat_session_group = pro_model.start_chat(history=history_to_pass_to_gemini_group)

            # Отправляем ввод пользователя
            loop = asyncio.get_running_loop()
            current_pro_response = await loop.run_in_executor(None, gemini_api.send_message_to_gemini, pro_model, pro_chat_session_group, triggered_pro_input)
            if not current_pro_response: raise ValueError("Initial Pro response None for group.")

            # Запускаем цикл FC
            final_pro_history_obj_list, _ = await process_gemini_fc_cycle(
                model_instance=pro_model, chat_session=pro_chat_session_group,
                available_functions_map=available_pro_functions,
                max_steps=max_pro_steps,
                original_chat_id=chat.id, original_user_id=user.id
            )

            # --- Извлечение и отправка ответа ---
            final_response_text = None # Сбрасываем перед извлечением
            if final_pro_history_obj_list:
                 last_model_content = final_pro_history_obj_list[-1]
                 if last_model_content.role == 'model':
                      text_parts = [part.text for part in last_model_content.parts if hasattr(part, 'text')]
                      if text_parts: final_response_text = "\n".join(text_parts).strip()
                      else: logger.info(f"Pro final response group {chat.id} no text part.")
                 else: logger.warning(f"Last history entry group {chat.id} not from model.")
            else: logger.error(f"Final pro history obj list empty/None group {chat.id}.")

            if final_response_text:
                 logger.info(f"Sending final Pro response group {chat.id}")
                 escaped_final_text = escape_markdown_v2(final_response_text)
                 try: await message.reply(escaped_final_text)
                 except Exception as send_err:
                      logger.error(f"Failed final reply group {chat.id}: {send_err}", exc_info=True)
                      try: await bot.send_message(chat.id, escaped_final_text)
                      except Exception as send_err2: logger.error(f"Failed final msg (no reply) group {chat.id}: {send_err2}", exc_info=True)
            elif not final_response_text and final_pro_history_obj_list:
                 logger.info(f"No final text send group {chat.id}, processing finished.")

            # --- Сохранение истории ---
            if final_pro_history_obj_list:
                 final_history_dict_list = gemini_history_to_dict_list(final_pro_history_obj_list)
                 # Используем длину ДО СТАРТА СЕССИИ для определения новых записей
                 new_history_entries = final_history_dict_list[original_history_len_for_save_group:]
                 logger.info(f"Attempting save {len(new_history_entries)} new history entries group {chat.id}.")
                 saved_count = 0
                 for entry_dict in new_history_entries:
                      role = entry_dict.get("role"); parts_list_dict = entry_dict.get("parts", [])
                      if not role or not parts_list_dict: continue
                      parts_to_save_dict = parts_list_dict
                      if role == "model": parts_to_save_dict = [p_dict for p_dict in parts_list_dict if 'function_call' not in p_dict and 'function_response' not in p_dict]
                      if parts_to_save_dict:
                           try:
                               await database.add_message_to_history(chat.id, role, parts_to_save_dict)
                               saved_count += 1
                           except Exception as db_save_err: logger.error(f"Failed save history entry (role: {role}) group {chat.id}: {db_save_err}", exc_info=True)
                 logger.info(f"Saved {saved_count} new entries group {chat.id}.")
            else:
                logger.warning(f"Final pro history obj list empty/None, nothing save group {chat.id}.")

        except Exception as e:
            logger.error(f"Error during Pro processing group {chat.id}: {e}", exc_info=True)

    end_time = time.monotonic()
    logger.info(f"Finished processing GROUP msg {chat.id} in {end_time - start_time:.2f} sec.")