# ./gemini_api.py

import google.generativeai as genai
import os
import json
import traceback
import logging
from typing import Optional # <--- ДОБАВЛЕН ЭТОТ ИМПОРТ

# --- ИМПОРТ ТИПОВ ИЗ google.ai.generativelanguage (КАК В РАБОЧЕМ ПРИМЕРЕ) ---
try:
    from google.ai import generativelanguage as glm
    # Определяем нужные типы через алиас glm
    Part = glm.Part
    FunctionResponse = glm.FunctionResponse
    FunctionDeclaration = glm.FunctionDeclaration
    Tool = glm.Tool
    Schema = glm.Schema
    Type = glm.Type
    # GenerationConfig все еще импортируется из genai.types
    GenerationConfig = genai.types.GenerationConfig
    # FinishReason может быть строкой, как мы выяснили, но проверим glm на всякий случай
    # Если его нет, будем использовать строки
    try:
        # Попробуем достать Enum из нужного места (может быть в Candidate)
        FinishReason = glm.Candidate.FinishReason
        print("DEBUG: Enum FinishReason успешно импортирован из glm.Candidate.")
    except AttributeError:
        print("Предупреждение: Не удалось импортировать glm.Candidate.FinishReason. Будет использоваться строковое сравнение.")
        FinishReason = None # Указываем, что Enum не найден

except ImportError:
    print("КРИТИЧЕСКАЯ ОШИБКА: Не удалось импортировать базовые типы из google.ai.generativelanguage.")
    print("Убедитесь, что у вас установлены необходимые зависимости (возможно, google-cloud-aiplatform?).")
    # Устанавливаем в None, чтобы инициализация не упала, но работа будет некорректной
    Part, FunctionResponse, FunctionDeclaration, Tool, Schema, Type, GenerationConfig, FinishReason = None, None, None, None, None, None, None, None
# --- КОНЕЦ ИМПОРТА ТИПОВ ---

# --- Инициализация логгера (если он еще не инициализирован глобально) ---
logger = logging.getLogger(__name__) # Именной логгер для этого модуля

# --- Вспомогательный импорт config ---
# Используем try-except на случай, если этот файл используется отдельно
try:
    import config
except ImportError:
    # Создаем заглушку, если config.py не используется в этом контексте
    class MockConfig:
        ENABLE_FUNCTION_CALLING = True # По умолчанию включено
        GENERATION_CONFIG = {"temperature": 0.7} # Пример конфига
        SAFETY_SETTINGS = [] # Пустые настройки
    config = MockConfig()
    logger.warning("Файл config.py не найден, используются настройки по умолчанию для gemini_api.")
# --- Конец импорта config ---


# --- Функция setup_gemini_model с использованием типов из glm и добавленным логгированием промпта ---
def setup_gemini_model(api_key: str, function_declarations_data: list, model_name: str, system_prompt: Optional[str]):
    """
    Настраивает и возвращает модель Gemini с инструментами и системным промптом.
    Использует типы из google.ai.generativelanguage.

    Args:
        api_key: API ключ Google AI.
        function_declarations_data: Список словарей с декларациями функций.
        model_name: Имя модели Gemini.
        system_prompt: Текст системного промпта (может быть None).

    Returns:
        Инициализированный объект genai.GenerativeModel.

    Raises:
        ImportError: Если необходимые типы не импортированы.
        ConnectionError: Если не удалось инициализировать модель.
        ValueError: При других ошибках конфигурации.
    """
    # Проверяем успешность импорта базовых типов
    if not Tool or not FunctionDeclaration or not Schema or not Type or not GenerationConfig:
         logger.critical("Не удалось импортировать необходимые типы (Tool, FunctionDeclaration, Schema, Type, GenerationConfig).")
         raise ImportError("Не удалось импортировать необходимые типы (Tool, FunctionDeclaration, Schema, Type, GenerationConfig).")

    genai.configure(api_key=api_key)
    tool_object = None # Инициализируем

    # Создаем инструменты только если включено в конфиге и есть декларации
    if function_declarations_data and config.ENABLE_FUNCTION_CALLING:
        logger.info("Создание конфигурации инструментов (из glm)...")
        declarations = []
        for func_decl_dict in function_declarations_data:
             # Базовая проверка валидности словаря декларации
             if not isinstance(func_decl_dict, dict) or 'name' not in func_decl_dict or 'description' not in func_decl_dict:
                  logger.warning(f"Пропуск неполной или некорректной декларации функции: {func_decl_dict}")
                  continue
             try:
                  param_schema = None
                  parameters_dict = func_decl_dict.get('parameters', {})
                  # Убедимся, что parameters_dict это словарь
                  if not isinstance(parameters_dict, dict):
                       logger.warning(f"'parameters' для функции {func_decl_dict['name']} не является словарем. Параметры пропущены.")
                       properties_dict = {}
                  else:
                       properties_dict = parameters_dict.get('properties', {})
                       if not isinstance(properties_dict, dict):
                            logger.warning(f"'properties' для функции {func_decl_dict['name']} не является словарем. Параметры пропущены.")
                            properties_dict = {}

                  # Создаем схему параметров, только если есть свойства
                  if properties_dict:
                      param_properties = {}
                      for param_name, param_details in properties_dict.items():
                           if not isinstance(param_details, dict):
                                logger.warning(f"Описание параметра '{param_name}' для функции {func_decl_dict['name']} не является словарем. Пропуск параметра.")
                                continue
                           param_type_str = param_details.get('type', 'STRING').upper() # STRING по умолчанию
                           try:
                               # Используем Type[...] для доступа к enum по строке
                               schema_type = Type[param_type_str]
                           except KeyError:
                                logger.warning(f"Неизвестный тип '{param_type_str}' для параметра '{param_name}'. Используется STRING.")
                                schema_type = Type.STRING
                           param_properties[param_name] = Schema(type=schema_type, description=param_details.get('description', ''))

                      # Получаем список обязательных полей
                      required_params = parameters_dict.get('required', [])
                      if not isinstance(required_params, list):
                           logger.warning(f"'required' для функции {func_decl_dict['name']} не является списком. Игнорируется.")
                           required_params = []

                      param_schema = Schema(
                           type=Type.OBJECT,
                           properties=param_properties,
                           required=required_params
                      )

                  # Создаем FunctionDeclaration с использованием импортированного типа
                  declarations.append(FunctionDeclaration(
                       name=func_decl_dict['name'],
                       description=func_decl_dict['description'],
                       parameters=param_schema # Будет None если нет параметров
                  ))
             except Exception as e:
                  logger.error(f"Ошибка при создании FunctionDeclaration для {func_decl_dict.get('name', 'UNKNOWN')}: {e}", exc_info=True)

        if declarations:
             # Создаем Tool с использованием импортированного типа
             tool_object = Tool(function_declarations=declarations)
             logger.info(f"Объект Tool успешно создан с {len(declarations)} декларациями.")
        else:
             logger.warning("Не удалось создать валидные декларации функций. Function Calling будет недоступен.")

    logger.info(f"Инициализация модели: {model_name}")
    try:
        # --- Используем generation_config и safety_settings из config ---
        gen_conf = config.GENERATION_CONFIG if isinstance(config.GENERATION_CONFIG, dict) else {}
        safe_set = config.SAFETY_SETTINGS if isinstance(config.SAFETY_SETTINGS, list) else []

        # Создаем объект GenerationConfig из словаря
        gen_config_obj = GenerationConfig(**gen_conf)

        init_args = {
            "model_name": model_name,
            "generation_config": gen_config_obj, # Передаем объект GenerationConfig
            "safety_settings": safe_set,         # Передаем список
        }
        if tool_object:
            init_args["tools"] = [tool_object] # Передаем список с одним объектом Tool

        # --- БЛОК ДЛЯ SYSTEM PROMPT И ЛОГИРОВАНИЯ ---
        if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
            # Логируем начало промпта для отладки (уровень DEBUG)
            logger.info(f"DEBUG: Установка system_instruction для модели '{model_name}'. Превью:\n-------\n{system_prompt[:500].strip()}...\n-------")
            # Добавляем промпт в аргументы инициализации
            # Убедитесь, что имя параметра 'system_instruction' актуально для вашей версии google-generativeai SDK
            init_args["system_instruction"] = system_prompt
        elif system_prompt:
            logger.warning(f"Системный промпт для '{model_name}' предоставлен, но он пуст или имеет неверный тип ({type(system_prompt)}). Игнорируется.")
        else:
             logger.info(f"Системный промпт для '{model_name}' не предоставлен.")
        # --- КОНЕЦ БЛОКА ---

        model = genai.GenerativeModel(**init_args)
        logger.info(f"Модель Gemini '{model_name}' успешно инициализирована.")
        return model

    except TypeError as te:
         logger.critical(f"TypeError при инициализации модели Gemini '{model_name}': {te}", exc_info=True)
         logger.critical("Проверьте формат generation_config, safety_settings, system_instruction и tools в соответствии с SDK.")
         raise ConnectionError(f"TypeError при инициализации Gemini: {te}") from te
    except Exception as e:
         logger.critical(f"Неожиданная ошибка инициализации модели Gemini '{model_name}': {e}", exc_info=True)
         raise ConnectionError(f"Не удалось инициализировать модель Gemini: {e}") from e


# --- Функция send_message_to_gemini ---
def send_message_to_gemini(model, chat, user_message):
    """
    Отправляет сообщение (строку, Part или список Part) в чат Gemini.
    """
    # Проверяем импорт Part
    if not Part:
        logger.critical("Критическая ошибка: Тип 'Part' не импортирован.")
        return None
    try:
        message_to_send = None
        if isinstance(user_message, str):
            message_to_send = user_message # Отправляем строку как есть
        elif isinstance(user_message, list) and all(isinstance(p, Part) for p in user_message):
            message_to_send = user_message # Отправляем список Part
        elif isinstance(user_message, Part):
            message_to_send = user_message # Отправляем один Part
        else:
             logger.warning(f"Неожиданный тип сообщения для send_message: {type(user_message)}. Попытка преобразовать в строку.")
             message_to_send = str(user_message)

        logger.debug(f"DEBUG: Отправка в chat.send_message типа: {type(message_to_send)}")
        if isinstance(message_to_send, list):
             logger.debug(f"DEBUG: Количество Part в списке: {len(message_to_send)}")

        response = chat.send_message(message_to_send)
        return response
    except IndexError as ie: # Ловим конкретно IndexError
        logger.error(f"Ошибка при доступе к кандидатам Gemini API: {ie}", exc_info=True)
        try:
            # Попытка получить фидбек, если кандидатов нет
            prompt_feedback = getattr(chat._history[-1], 'prompt_feedback', None) # Доступ к истории может быть другим!
            if prompt_feedback:
                logger.warning(f"Gemini API Prompt Feedback (возможно, причина блока): {prompt_feedback}")
        except Exception as pf_err:
            logger.error(f"Не удалось получить prompt_feedback: {pf_err}")
        return None # Возвращаем None при ошибке
    except Exception as e:
        logger.error(f"Ошибка при отправке сообщения в Gemini API: {e}", exc_info=True)
        # traceback.print_exc() # Логируется через exc_info=True
        return None


# --- Функция handle_function_call ---
# (остальной код файла без изменений)
def handle_function_call(response, available_functions, chat):
    """
    Обрабатывает ответ Gemini, находит ВСЕ вызовы функций, выполняет их
    и отправляет СПИСОК ответов обратно.
    """
    # Проверяем импорт необходимых типов
    if not Part or not FunctionResponse:
         logger.error("Ошибка: Отсутствуют типы Part или FunctionResponse (не удалось импортировать).")
         return None

    try:
        if not response or not response.candidates:
            logger.warning("В ответе Gemini нет кандидатов или ответ пуст.")
            return None
        candidate = response.candidates[0]

        # Проверка безопасности
        is_safe = True
        if hasattr(candidate, 'finish_reason'):
             reason = candidate.finish_reason
             # Сначала пробуем сравнить с Enum, если он импортировался
             if FinishReason and reason == FinishReason.SAFETY: is_safe = False
             elif isinstance(reason, str) and reason.upper() == "SAFETY": is_safe = False
             elif FinishReason and reason == FinishReason.RECITATION: is_safe = False
             elif isinstance(reason, str) and reason.upper() == "RECITATION": is_safe = False

             if not is_safe:
                  ratings = getattr(candidate, 'safety_ratings', 'N/A')
                  logger.warning(f"Ответ заблокирован (FinishReason={reason}). Рейтинги: {ratings}")
                  return None # Возвращаем None при блокировке

        # Проверка наличия контента и частей
        if not candidate.content or not candidate.content.parts:
            reason_str = getattr(candidate, 'finish_reason', 'UNKNOWN')
            logger.warning(f"В ответе Gemini нет контента/частей. Причина: {reason_str}")
            return None # Нечего обрабатывать

        # --- Находим ВСЕ function calls ---
        function_calls_to_process = []
        logger.debug("Поиск function_call в частях ответа...")
        for i, part in enumerate(candidate.content.parts):
            if hasattr(part, 'function_call') and part.function_call:
                fc_name = getattr(part.function_call, 'name', '')
                logger.debug(f"НАЙДЕН function_call в части {i}: Name='{fc_name}'")
                # Убедимся, что это действительно объект вызова функции С ИМЕНЕМ
                if fc_name and hasattr(part.function_call, 'args'):
                     function_calls_to_process.append(part.function_call)
                else:
                     logger.warning(f"Найден объект function_call, но он не имеет имени или ожидаемой структуры в части {i}.")

        # Если не было найдено валидных вызовов функций
        if not function_calls_to_process:
            logger.debug("Function calls не найдены или невалидны.")
            return None # Возвращаем None, т.к. FC не было

        # --- Обрабатываем КАЖДЫЙ найденный вызов и собираем ответы ---
        function_response_parts = [] # Список для хранения Part(FunctionResponse)
        logger.info(f"--- Обнаружено {len(function_calls_to_process)} запросов на вызов функций ---")

        for i, fc in enumerate(function_calls_to_process):
            logger.debug(f"Начало обработки FC #{i+1}/{len(function_calls_to_process)}")
            function_name = fc.name
            # Преобразуем аргументы из proto Map в обычный dict
            args = dict(fc.args) if fc.args else {}
            logger.info(f"  Обработка вызова: {function_name}({args})")

            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                try:
                    # Вызываем соответствующую Python-функцию
                    function_response_data = function_to_call(**args)
                    # Ограничиваем вывод результата для логов
                    result_preview = str(function_response_data)[:200]
                    if len(str(function_response_data)) > 200: result_preview += "..."
                    logger.info(f"  Результат '{function_name}': {result_preview}")

                    # --- ВАЖНО: Упаковываем результат в словарь, как ожидает FunctionResponse ---
                    if not isinstance(function_response_data, dict):
                        logger.warning(f"Функция {function_name} вернула не словарь ({type(function_response_data)}). Оборачиваем в {{'result': ...}}")
                        function_response_data = {"result": str(function_response_data)}

                except Exception as e:
                    logger.error(f"  Ошибка при выполнении функции '{function_name}': {e}", exc_info=True)
                    # traceback.print_exc() # Уже в логах
                    function_response_data = {"error": f"Error executing function '{function_name}': {e}"}

                # Создаем Part с FunctionResponse для этого вызова
                try:
                     response_part = Part(
                         function_response=FunctionResponse(
                             name=function_name,
                             response=function_response_data # Передаем словарь
                         )
                     )
                     function_response_parts.append(response_part)
                     logger.debug(f"Добавлен ответ для FC #{i+1}")
                except Exception as fr_e:
                     # Ошибка при создании FunctionResponse (например, если data не сериализуется)
                     logger.error(f"  КРИТИЧЕСКАЯ ОШИБКА при создании FunctionResponse для '{function_name}': {fr_e}", exc_info=True)
                     # traceback.print_exc() # Уже в логах
                     # Добавляем ответ об ошибке, чтобы не нарушать количество
                     error_response_part = Part(
                          function_response=FunctionResponse(
                              name=function_name,
                              response={"error": f"Failed to create response object for {function_name}: {fr_e}"}
                          )
                     )
                     function_response_parts.append(error_response_part)

            else:
                # Обработка неизвестной функции
                logger.error(f"  Ошибка: Модель запросила неизвестную функцию: {function_name}")
                error_response_part = Part(
                     function_response=FunctionResponse(
                         name=function_name,
                         response={"error": f"Function '{function_name}' is not available."}
                     )
                )
                function_response_parts.append(error_response_part)
                logger.debug(f"Добавлен ответ об ошибке для неизвестной функции #{i+1}")

            logger.debug(f"Завершение обработки FC #{i+1}. Всего готово ответов: {len(function_response_parts)}")
        # --- КОНЕЦ ЦИКЛА ОБРАБОТКИ FC ---

        # --- Отправляем СПИСОК ответов обратно ---
        if function_response_parts:
             # Убедимся, что количество собранных ответов совпадает с количеством запросов
            if len(function_response_parts) != len(function_calls_to_process):
                 logger.critical(f"КРИТИЧЕСКОЕ РАСХОЖДЕНИЕ: Запрошено {len(function_calls_to_process)} FC, но подготовлено {len(function_response_parts)} ответов!")
                 return None # Не отправляем ничего, чтобы не вызвать ошибку 400

            logger.info(f"Отправка {len(function_response_parts)} ответов функций обратно в Gemini...")
            # Передаем СПИСОК объектов Part в send_message_to_gemini
            final_response = send_message_to_gemini(model=None, chat=chat, user_message=function_response_parts)
            logger.debug(f"Результат send_message_to_gemini (список ответов): {'Успешно (получен ответ)' if final_response else 'Неудачно (API вернуло None или ошибка)'}")
            return final_response # Возвращаем ответ модели ПОСЛЕ обработки FC
        else:
             logger.error("Ошибка: Не удалось сформировать ответы на вызовы функций (function_response_parts пуст).")
             return None

    except AttributeError as e:
         logger.error(f"Ошибка при доступе к атрибутам ответа Gemini: {e}. Ответ: {response}", exc_info=True)
         # traceback.print_exc() # Уже в логах
         return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при обработке ответа Gemini или вызове функции: {e}", exc_info=True)
        # traceback.print_exc() # Уже в логах
        return None