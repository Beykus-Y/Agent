# handlers/private.py
import time
import logging
from typing import Dict, Any, Callable

# --- Aiogram Imports ---
from aiogram import types, F
from aiogram.enums import ChatType

# --- Локальные импорты ---
import database
from bot_loader import dp, bot # Нужен bot для отправки ошибок, если reply не работает
from handlers.common import ( # Импортируем общие функции
    prepare_gemini_session_history,
    run_gemini_interaction,
    extract_final_text,
    send_final_response,
    save_new_history_entries
)
from bot_utils import escape_markdown_v2

logger = logging.getLogger(__name__)

@dp.message(F.text & (F.chat.type == ChatType.PRIVATE))
async def handle_private_message(message: types.Message,
                                 pro_model: Any, # Получаем из dp.workflow_data
                                 available_pro_functions: Dict[str, Callable], # Из dp.workflow_data
                                 max_pro_steps: int): # Из dp.workflow_data
    """Обработчик для личных сообщений."""
    user = message.from_user
    chat = message.chat
    user_input = message.text or ""
    start_time = time.monotonic()

    if not user:
        logger.warning("Private message without user info?")
        return
    logger.info(f"PRIVATE Msg from user={user.id} chat={chat.id}")

    # Обновляем профиль пользователя
    try:
        await database.upsert_user_profile(user.id, user.username, user.first_name, user.last_name)
    except Exception as db_err:
        logger.error(f"Upsert profile failed user {user.id}: {db_err}", exc_info=True)

    # Проверка наличия Pro модели
    if not pro_model:
        logger.critical("Pro model missing in workflow_data!")
        await message.reply(escape_markdown_v2("AI model is unavailable. Please try again later."))
        return

    # Основной блок обработки
    try:
        # 1. Подготовка истории
        initial_history, original_len = await prepare_gemini_session_history(chat.id, user.id, add_notes=True)

        # 2. Взаимодействие с Gemini
        # --- Получаем имя последней функции ---
        final_history_obj_list, error_msg, last_func = await run_gemini_interaction(
            model_instance=pro_model,
            initial_history=initial_history,
            user_input=user_input,
            available_functions=available_pro_functions,
            max_steps=max_pro_steps,
            chat_id=chat.id,
            user_id=user.id
        )

        # 3. Обработка результата взаимодействия
        if error_msg:
            # Если run_gemini_interaction вернул ошибку, сообщаем пользователю
            logger.error(f"Gemini interaction failed for private chat {chat.id}: {error_msg}")
            await message.reply(escape_markdown_v2(f"Error: {error_msg}"))
            # История не сохраняется в этом случае, т.к. final_history_obj_list будет None
        elif final_history_obj_list:
            # --- ИЗМЕНЕНИЕ: Условие для отправки финального ответа ---
            if last_func != 'send_telegram_message':
                # 4. Извлечение и отправка ответа
                final_response_text = extract_final_text(final_history_obj_list)
                await send_final_response(message, final_response_text)
            else:
                 logger.info(f"Final text response suppressed because last function call was 'send_telegram_message' in chat {chat.id}")

            # 5. Сохранение истории (только если взаимодействие прошло успешно)
            await save_new_history_entries(chat.id, final_history_obj_list, original_len, current_user_id=user.id)

    except Exception as e:
        # Ловим остальные неожиданные ошибки
        logger.error(f"Unhandled error processing private message chat {chat.id}: {e}", exc_info=True)
        try:
            error_msg_escaped = escape_markdown_v2(f"An internal error occurred processing your message.")
            await message.reply(error_msg_escaped)
        except Exception as send_err:
            logger.error(f"Failed send error reply chat {chat.id}: {send_err}", exc_info=True)

    end_time = time.monotonic()
    logger.info(f"Finished processing PRIVATE msg {chat.id} in {end_time - start_time:.2f} sec.")