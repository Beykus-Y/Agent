# handlers/group.py
import time
import logging
import asyncio
import json # Для логирования ответов
from typing import Dict, Any, Callable, Optional, List

# --- Aiogram Imports ---
from aiogram import types, F
from aiogram.enums import ChatType

# --- Локальные импорты ---
import database
from bot_loader import dp, bot
from handlers.common import (
    prepare_gemini_session_history,
    run_gemini_interaction,
    extract_final_text,
    send_final_response,
    save_new_history_entries
)
# process_gemini_fc_cycle больше не нужен здесь напрямую
# import gemini_api # Нужен для вызова send_message_to_gemini напрямую, если бы мы его вызывали из Lite
from bot_utils import escape_markdown_v2
# --- Импорт обработчика remember_user_info ---
# Убедитесь, что путь импорта правильный
from tool_handlers import remember_user_info

# --- Типы Google ---
try:
    from google.ai import generativelanguage as glm
    Content = glm.Content
except ImportError:
    logging.critical("Failed to import google types (glm) in handlers.group")
    Content = None # type: ignore

# --- Инициализация логгера ---
logger = logging.getLogger(__name__)

# handlers/group.py
import time
import logging
import asyncio
import json # Для логирования ответов
from typing import Dict, Any, Callable, Optional, List

# --- Aiogram Imports ---
from aiogram import types, F
from aiogram.enums import ChatType

# --- Локальные импорты ---
import database
from bot_loader import dp, bot
from handlers.common import (
    prepare_gemini_session_history,
    run_gemini_interaction,
    extract_final_text,
    send_final_response,
    save_new_history_entries
)
# --- ИСПРАВЛЕНО: Убран ненужный импорт ---
# import gemini_api
from bot_utils import escape_markdown_v2
# --- Импорт обработчика remember_user_info ---
from tool_handlers import remember_user_info

# --- Типы Google ---
try:
    from google.ai import generativelanguage as glm
    Content = glm.Content
except ImportError:
    logging.critical("Failed to import google types (glm) in handlers.group")
    Content = None # type: ignore

# --- Инициализация логгера ---
logger = logging.getLogger(__name__)

@dp.message(F.text & (F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP})))
async def handle_group_message(message: types.Message,
                               lite_model: Any,
                               pro_model: Any,
                               available_lite_functions: Dict[str, Callable], # Нужны только для setup
                               available_pro_functions: Dict[str, Callable],
                               max_lite_steps: int, # Не используется в текущей логике
                               max_pro_steps: int):
    """Обработчик для сообщений в группах."""
    user = message.from_user
    chat = message.chat
    user_input = message.text or ""
    start_time = time.monotonic()

    if not user:
        logger.warning("Group message without user info?")
        return
    logger.info(f"GROUP Msg from user={user.id} chat={chat.id} message_id={message.message_id}")

    try:
        # Конвертируем текст в формат parts для БД
        user_parts = [{"text": user_input}]
        await database.add_message_to_history(chat.id, 'user', user_parts, user_id=user.id)
        logger.debug(f"Saved user message (id={message.message_id}) to history before Lite analysis.")
    except Exception as initial_save_err:
        logger.error(f"Failed to save initial user message to history chat {chat.id}: {initial_save_err}", exc_info=True)

    try:
        await database.upsert_user_profile(user.id, user.username, user.first_name, user.last_name)
    except Exception as db_err:
        logger.error(f"Upsert profile failed user {user.id} in group {chat.id}: {db_err}", exc_info=True)

    # --- Логика анализа сообщения Lite моделью ---
    run_pro_model = False
    pro_triggered_by_lite = False
    triggered_pro_input = user_input
    lite_requested_no_action = False

    if not lite_model:
        logger.error(f"Lite model missing! Cannot analyze group message for chat {chat.id}.")
    else:
        try:
            logger.info(f"Running Lite model analysis for group chat {chat.id}")
            # 1. Подготовка истории для Lite
            lite_history_dict, lite_original_len = await prepare_gemini_session_history(chat.id, user.id, add_notes=False)

            # 2. Запуск сессии и отправка сообщения в Lite модель
            lite_chat_session = lite_model.start_chat(history=lite_history_dict)
            loop = asyncio.get_running_loop()
            # --- ИСПРАВЛЕНО: Импортируем gemini_api для вызова ---
            import gemini_api # Импорт нужен здесь, если он не глобальный
            lite_initial_response = await loop.run_in_executor(
                None, gemini_api.send_message_to_gemini, lite_model, lite_chat_session, user_input
            )

            if lite_initial_response is None:
                logger.error(f"Initial Lite model response was None for group chat {chat.id}.")
            elif Content is None:
                 logger.critical("Google type 'Content' not imported. Cannot process Lite response.")
            else:
                # --- ЛОГИРОВАНИЕ ОТВЕТА LITE ---
                try:
                    response_parts_log = []
                    finish_reason_log = "N/A"
                    safety_ratings_log = "N/A"
                    if hasattr(lite_initial_response, 'candidates') and lite_initial_response.candidates:
                        candidate = lite_initial_response.candidates[0]
                        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                             for i, part in enumerate(candidate.content.parts):
                                  part_info = f"Part {i}: Type={type(part)}"
                                  if hasattr(part, 'text') and part.text is not None:
                                       part_info += f", Text='{part.text[:100]}...'"
                                  if hasattr(part, 'function_call') and part.function_call is not None:
                                       fc_name = getattr(part.function_call, 'name', 'N/A')
                                       part_info += f", FunctionCall(Name='{fc_name}')"
                                  response_parts_log.append(part_info)
                        finish_reason_log = f"Finish Reason: {getattr(candidate, 'finish_reason', 'N/A')}"
                        safety_ratings_log = f"Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}"
                    else:
                         finish_reason_log = f"Finish Reason: {getattr(lite_initial_response, 'prompt_feedback', 'N/A')}"

                    logger.info(f"Lite model RAW response content for chat {chat.id}: Parts=[{'; '.join(response_parts_log)}], {finish_reason_log}, {safety_ratings_log}")
                except Exception as log_err:
                    logger.error(f"Error logging Lite model response details for chat {chat.id}: {log_err}", exc_info=True)
                # --- КОНЕЦ ЛОГИРОВАНИЯ ---

                # --- ИСПРАВЛЕННЫЙ БЛОК АНАЛИЗА LITE ---
                # 3. Анализ ВСЕХ Function Calls в ПЕРВОМ ответе Lite модели
                calls_found: Dict[str, List[Dict[str, Any]]] = {"trigger_pro_model_processing": [], "remember_user_info": []}
                triggered_fc_args = None
                lite_response_text = None

                if hasattr(lite_initial_response, 'candidates') and lite_initial_response.candidates:
                    candidate = lite_initial_response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                         # Сначала ищем текст и FC
                         for part in candidate.content.parts:
                              # Ищем текст
                              if hasattr(part, 'text') and part.text:
                                  lite_response_text = part.text.strip()
                                  logger.debug(f"Lite response text found: '{lite_response_text}'")
                              # Ищем FC
                              if hasattr(part, 'function_call') and part.function_call:
                                   fc_name = getattr(part.function_call, 'name', None)
                                   if fc_name in calls_found:
                                        logger.info(f"Lite model requested '{fc_name}' in initial response for chat {chat.id}")
                                        fc_args = {}
                                        if hasattr(part.function_call, 'args'):
                                             try: fc_args = dict(part.function_call.args)
                                             except Exception as args_err: logger.error(f"Failed to parse args for Lite FC '{fc_name}' chat {chat.id}: {args_err}")
                                        calls_found[fc_name].append(fc_args)
                                   elif fc_name: logger.warning(f"Lite model requested unexpected function '{fc_name}' chat {chat.id}")

                # Теперь определяем действия
                run_pro_model = False
                pro_triggered_by_lite = False

                # Приоритет у триггера Pro модели
                if calls_found["trigger_pro_model_processing"]:
                    pro_triggered_by_lite = True
                    run_pro_model = True
                    logger.info(f"Lite model triggered Pro model processing for group chat {chat.id}")
                    if calls_found["trigger_pro_model_processing"]:
                        triggered_fc_args = calls_found["trigger_pro_model_processing"][0]
                    else: triggered_fc_args = None

                    if triggered_fc_args:
                        if 'user_input' in triggered_fc_args:
                            triggered_pro_input = triggered_fc_args['user_input']
                            logger.info(f"Using user_input from Lite's FunctionCall args for chat {chat.id}")
                        else: logger.warning(f"Lite triggered Pro, but 'user_input' missing in FC args chat {chat.id}. Using original.")
                    else: logger.warning(f"Lite triggered Pro, but failed to get FC args chat {chat.id}. Using original.")
                    logger.info(f"Ignoring {len(calls_found['remember_user_info'])} remember_user_info calls because Pro model was triggered.")

                # Если триггера Pro НЕ БЫЛО, но есть remember_user_info, выполняем его
                elif calls_found["remember_user_info"]:
                    logger.info(f"Lite model requested {len(calls_found['remember_user_info'])} remember_user_info calls. Executing them.")
                    executed_remembers = 0
                    for remember_args in calls_found["remember_user_info"]:
                        rem_category = remember_args.get('info_category'); rem_value = remember_args.get('info_value')
                        rem_user_id_from_args = remember_args.get('user_id')
                        correct_user_id = rem_user_id_from_args if isinstance(rem_user_id_from_args, int) else user.id
                        if correct_user_id and rem_category and rem_value:
                             try:
                                  result = await remember_user_info(user_id=correct_user_id, info_category=rem_category, info_value=rem_value)
                                  if result.get("status") == "success": executed_remembers += 1
                                  else: logger.error(f"remember_user_info handler failed for args {remember_args}: {result.get('message')}")
                             except Exception as rem_exec_err: logger.error(f"Error executing remember_user_info handler for args {remember_args}: {rem_exec_err}", exc_info=True)
                        else: logger.warning(f"Skipping remember_user_info call due to missing args: {remember_args} (Used correct user_id: {correct_user_id})")
                    logger.info(f"Executed {executed_remembers} remember_user_info calls initiated by Lite model for chat {chat.id}.")

                elif lite_response_text == "NO_ACTION_NEEDED":
                    lite_requested_no_action = True
                    logger.info(f"Lite model explicitly requested no action via marker for group chat {chat.id}")

                # 4. --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
                # Если НЕ было FC и текст НЕ маркер (и текст вообще есть)
                elif lite_response_text:
                    logger.warning(f"Lite model generated unexpected text instead of FC or marker: '{lite_response_text[:100]}...'. IMPLICITLY triggering Pro model for chat {chat.id}.")
                    # Считаем это неявным триггером для Pro
                    pro_triggered_by_lite = True # Помечаем, что это инициировано Lite (хоть и неявно)
                    run_pro_model = True
                    # Используем оригинальный ввод пользователя, так как Lite не передала аргументы
                    triggered_pro_input = user_input
                # ---------------------------

                # 5. Если не было ни FC, ни текста (теоретически возможно)
                else:
                    logger.info(f"Lite model finished without FC calls and without any text response for group chat {chat.id}. Assuming no action.")
                    lite_requested_no_action = True # Считаем, что действий не требуется

        except Exception as lite_err:
            logger.error(f"Error during Lite model analysis for group {chat.id}: {lite_err}", exc_info=True)
            if not pro_triggered_by_lite: # Если не было явного триггера до ошибки
                run_pro_model = False

    # Запуск Pro модели, если требуется
    if run_pro_model:
        logger.info(f"Running Pro model for group chat={chat.id}, triggered by user={user.id}")
        if not pro_model:
            logger.error(f"Pro model missing for group processing! Chat ID: {chat.id}")
        else:
            # --- ИСПРАВЛЕНА СТРУКТУРА TRY...EXCEPT ---
            try:
                # 1. Подготовка истории Pro
                initial_history_pro, original_len_pro = await prepare_gemini_session_history(chat.id, user.id, add_notes=True)

                # 2. Взаимодействие с Gemini Pro
                final_history_obj_list_pro, error_msg_pro, last_func_pro = await run_gemini_interaction(
                    model_instance=pro_model,
                    initial_history=initial_history_pro,
                    user_input=triggered_pro_input,
                    available_functions=available_pro_functions,
                    max_steps=max_pro_steps,
                    chat_id=chat.id,
                    user_id=user.id
                )

                # 3. Обработка результата взаимодействия Pro
                if error_msg_pro:
                    logger.error(f"Gemini Pro interaction failed for group chat {chat.id}: {error_msg_pro}")
                    try: await message.reply(escape_markdown_v2(f"Произошла ошибка при обработке вашего запроса: {error_msg_pro}"))
                    except Exception as reply_err: logger.error(f"Failed to send error reply in group {chat.id}: {reply_err}")
                elif final_history_obj_list_pro:
                    # Условие для отправки финального ответа
                    if last_func_pro != 'send_telegram_message':
                        # 4. Извлечение и отправка ответа Pro
                        final_response_text_pro = extract_final_text(final_history_obj_list_pro)
                        await send_final_response(message, final_response_text_pro)
                    else:
                        logger.info(f"Final text response suppressed because last function call was 'send_telegram_message' in chat {chat.id}")

                    # 5. Сохранение истории Pro модели
                    await save_new_history_entries(chat.id, final_history_obj_list_pro, original_len_pro, current_user_id=user.id)
                else:
                    # Случай, когда нет ошибки, но и истории нет
                    logger.error(f"Gemini Pro interaction returned empty history without error message for chat {chat.id}")
                    try: await message.reply(escape_markdown_v2("Не удалось получить ответ от ИИ после обработки."))
                    except Exception as reply_err: logger.error(f"Failed to send empty history error reply in group {chat.id}: {reply_err}")

            except Exception as e: # Этот except ловит ошибки из блока try выше
                logger.error(f"Unhandled error processing Pro model for group chat {chat.id}: {e}", exc_info=True)
                try:
                    await message.reply(escape_markdown_v2("Произошла внутренняя ошибка при обработке вашего запроса."))
                except Exception as reply_err:
                    logger.error(f"Failed to send internal error reply in group {chat.id}: {reply_err}")
            # --- КОНЕЦ ИСПРАВЛЕНИЯ СТРУКТУРЫ ---

    # Если run_pro_model остался False после анализа Lite, бот просто ничего не ответит.

    end_time = time.monotonic()
    logger.info(f"Finished processing GROUP msg {chat.id} (msg_id={message.message_id}) in {end_time - start_time:.2f} sec.")

    # Если run_pro_model остался False после анализа Lite, бот просто ничего не ответит.
