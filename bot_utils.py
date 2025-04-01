# bot_utils.py
import re
import logging
from typing import Dict, Any, List, Optional

# --- Типы Google (только для аннотаций) ---
# Это помогает избежать циклического импорта, если типы нужны в других местах
try:
    from google.ai.generativelanguage import Part, Content, FunctionResponse, FunctionCall
except ImportError:
    # Определяем заглушки, если импорт не удался
    Part = Any
    Content = Any
    FunctionResponse = Any
    FunctionCall = Any

logger = logging.getLogger(__name__)

# --- Утилита экранирования Markdown V2 ---
def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для Telegram MarkdownV2."""
    if not isinstance(text, str):
        return ""
    # Список символов для экранирования: _ * [ ] ( ) ~ ` > # + - = | { } . !
    # Важно: экранируем \ самим собой тоже, если он не часть escape-последовательности
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    # Экранируем все символы из списка
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)


# --- Конвертеры истории Gemini ---
def _convert_part_to_dict(part: Part) -> Optional[Dict[str, Any]]:
    """Преобразует объект Part (или dict) в словарь для сохранения или логирования."""
    part_dict = {}
    try:
        if hasattr(part, 'text') and part.text: # Сохраняем только непустой текст
            part_dict['text'] = part.text

        # Обработка FunctionCall
        if hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            fc_data = {'name': getattr(fc, 'name', None)}
            # --- ИСПРАВЛЕНИЕ TypeError: 'NoneType' object is not iterable ---
            fc_args = getattr(fc, 'args', None) # Получаем args
            if fc_args is not None: # Проверяем, что args не None
                try:
                    fc_data['args'] = dict(fc_args) # Конвертируем в dict только если не None
                except TypeError as e:
                    logger.error(f"Could not convert function_call args to dict for '{fc_data['name']}': {e}. Args: {fc_args}")
                    fc_data['args'] = {"error": "failed to parse args"} # Записываем ошибку
            else:
                 fc_data['args'] = {} # Если args был None или отсутствовал, ставим пустой dict
            part_dict['function_call'] = fc_data

        # Обработка FunctionResponse
        if hasattr(part, 'function_response') and part.function_response:
            fr = part.function_response
            part_dict['function_response'] = {
                'name': getattr(fr, 'name', None),
                'response': getattr(fr, 'response', {}) # 'response' уже должен быть dict
            }

        # Обработка inline_data (если нужно)
        # if hasattr(part, 'inline_data') and part.inline_data:
        #    # ... логика обработки inline_data ...
        #    pass

        # Возвращаем словарь только если он не пустой
        return part_dict if part_dict else None

    except Exception as e:
        logger.error(f"Error converting Part to dict: {e}. Part: {part}", exc_info=True)
        return {"error": f"Failed to process Part: {e}"}


def gemini_history_to_dict_list(history: List[Content]) -> List[Dict[str, Any]]:
    """Преобразует историю Gemini (список Content) в список словарей для БД."""
    dict_list = []
    if not history:
        return dict_list

    for entry in history:
        role = getattr(entry, 'role', None)
        if not role:
            logger.warning("History entry missing role.")
            continue

        parts_list_of_dicts = []
        if hasattr(entry, 'parts'):
            # Конвертируем каждую часть и фильтруем None (ошибки или пустые части)
            converted_parts = [_convert_part_to_dict(p) for p in entry.parts]
            parts_list_of_dicts = [p for p in converted_parts if p is not None]

        # Добавляем запись в итоговый список, только если есть роль и обработанные части
        if parts_list_of_dicts:
            dict_list.append({"role": role, "parts": parts_list_of_dicts})
        elif role: # Если роль есть, но части пустые/неконвертируемые - логируем
             logger.debug(f"History entry for role '{role}' resulted in empty parts list after conversion.")

    return dict_list