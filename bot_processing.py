# bot_processing.py
import asyncio
import inspect
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

# --- Локальные импорты ---
import tool_handlers
import gemini_api
from bot_utils import escape_markdown_v2

# --- Экземпляр Bot ---
from bot_loader import bot as aiogram_bot_instance

logger = logging.getLogger(__name__)

# --- Типы Google ---
try:
    from google.ai import generativelanguage as glm
    Content = glm.Content
    Part = glm.Part
    FunctionResponse = glm.FunctionResponse
    FunctionCall = glm.FunctionCall
    from google.generativeai.types import GenerateContentResponse
    logger.debug("Successfully imported types from glm and generativeai.types in bot_processing")
except ImportError as e:
    logger.critical(f"Failed to import google types in bot_processing: {e}", exc_info=True)
    Part, Content, FunctionResponse, FunctionCall, GenerateContentResponse = None, None, None, None, None
    exit(1)

# Функция execute_function_call остается без изменений

async def execute_function_call(
        handler_func: Callable,
        args: Dict[str, Any],
        chat_id_for_handlers: Optional[int] = None,
        user_id_for_handlers: Optional[int] = None
        ) -> Any:
    """
    Выполняет хендлер функции (синхронный или асинхронный),
    передавая ему аргументы из Gemini и опционально chat_id/user_id.
    """
    handler_sig = inspect.signature(handler_func)
    final_args = args.copy()

    if 'chat_id' in handler_sig.parameters:
        if chat_id_for_handlers is not None: final_args['chat_id'] = chat_id_for_handlers
        else: logger.warning(f"Handler '{handler_func.__name__}' expects 'chat_id', but it was not provided."); final_args.pop('chat_id', None)

    if 'user_id' in handler_sig.parameters:
        if user_id_for_handlers is not None: final_args['user_id'] = user_id_for_handlers
        else: logger.warning(f"Handler '{handler_func.__name__}' expects 'user_id', but it was not provided."); final_args.pop('user_id', None)

    filtered_args = {k: v for k, v in final_args.items() if k in handler_sig.parameters}

    missing_args = [
        p_name for p_name, p_obj in handler_sig.parameters.items()
        if p_obj.default is inspect.Parameter.empty and p_name not in filtered_args
    ]
    if missing_args:
         logger.error(f"Missing required arguments for '{handler_func.__name__}': {missing_args}. Provided args: {list(filtered_args.keys())}")
         return {"error": f"Missing required arguments for function '{handler_func.__name__}': {', '.join(missing_args)}"}

    try:
        if asyncio.iscoroutinefunction(handler_func):
            return await handler_func(**filtered_args)
        else:
            loop = asyncio.get_running_loop()
            from functools import partial
            func_call = partial(handler_func, **filtered_args)
            return await loop.run_in_executor(None, func_call)
    except Exception as exec_err:
        logger.error(f"Error executing handler '{handler_func.__name__}' with args {filtered_args}: {exec_err}", exc_info=True)
        return {"error": f"Handler execution failed for function '{handler_func.__name__}': {exec_err}"}


# --- ИЗМЕНЕНО: Возвращаемый тип Tuple теперь содержит имя последней функции ---
async def process_gemini_fc_cycle(
    model_instance: Any,
    chat_session: Any,
    available_functions_map: Dict[str, Callable],
    max_steps: int,
    original_chat_id: Optional[int] = None,
    original_user_id: Optional[int] = None,
) -> Tuple[Optional[List[Content]], Optional[str]]: # <--- ИЗМЕНЕН ТИП: Возвращает историю и имя последней функции
    """
    Обрабатывает цикл Function Calling для ответа Gemini.
    Отправляет ответы на ВСЕ FC одним запросом.
    Возвращает финальную историю и имя последней успешно вызванной функции.
    """
    if not Part or not FunctionResponse or not FunctionCall or not Content or not GenerateContentResponse:
         logger.critical("Missing Google types! Cannot process Function Calling.")
         return getattr(chat_session, 'history', None), None # <--- Возвращаем None для имени функции

    try:
        if not hasattr(chat_session, 'history') or not chat_session.history:
             logger.warning("Chat session history is empty or missing before FC cycle.")
             return getattr(chat_session, 'history', None), None # <--- Возвращаем None

        last_content = chat_session.history[-1]
        if not isinstance(last_content, Content):
             logger.error(f"Last history item is not Content object: {type(last_content)}")
             return chat_session.history, None # <--- Возвращаем None
        if last_content.role != 'model':
             logger.debug("Last message in history is not from model, no FC cycle needed.")
             return chat_session.history, None # <--- Возвращаем None

        class MockResponse:
            def __init__(self, content):
                class MockCandidate:
                    def __init__(self, content): self.content = content; self.safety_ratings = []; self.finish_reason = 1
                self.candidates = [MockCandidate(content)] if content else []
        current_response: Optional[GenerateContentResponse] = MockResponse(last_content)

    except Exception as e:
         logger.error(f"Failed get last response from session history: {e}", exc_info=True)
         return getattr(chat_session, 'history', None), None # <--- Возвращаем None

    # --- ИЗМЕНЕНО: Переменная для хранения имени последней успешной функции ---
    last_successful_fc_name: Optional[str] = None
    # -------------------------------------------------------------------
    step = 0
    while current_response and step < max_steps:
        step += 1
        model_name_str = getattr(model_instance, 'model_name', 'Unknown Model')
        logger.info(f"--- FC Analysis ({model_name_str} Step {step}/{max_steps}) ---")

        try:
            if not current_response.candidates: logger.info("No candidates in response."); break
            candidate = current_response.candidates[0]
            if not candidate.content or not candidate.content.parts:
                finish_reason = getattr(candidate, 'finish_reason', 'N/A')
                logger.info(f"No content/parts in candidate (Finish reason: {finish_reason}). Ending FC cycle.")
                break
            parts = candidate.content.parts
        except (IndexError, AttributeError, TypeError) as e:
            logger.warning(f"Response structure error accessing parts: {e}. Response: {current_response}")
            break

        function_calls_to_process: List[FunctionCall] = []
        for part in parts:
            if isinstance(part, Part) and hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                if isinstance(fc, FunctionCall) and hasattr(fc, 'name') and fc.name:
                    function_calls_to_process.append(fc)
                else:
                    logger.warning(f"Found part with 'function_call', but it's not a valid FunctionCall object or has no name: {fc}")

        if not function_calls_to_process:
            logger.info("No valid Function Calls found in this step.")
            break

        logger.info(f"Found {len(function_calls_to_process)} FCs by {model_name_str}.")

        response_parts_for_gemini: List[Part] = []
        fc_exec_tasks = []

        for fc in function_calls_to_process:
            function_name = fc.name
            args: Dict[str, Any] = {}
            if hasattr(fc, 'args'):
                 try: args = dict(fc.args)
                 except TypeError as e:
                      logger.error(f"Cannot convert args to dict for FC '{function_name}': {e}. Args: {fc.args}")
                      error_response = Part(function_response=FunctionResponse(name=function_name, response={"error": f"Failed to parse arguments: {e}"}))
                      response_parts_for_gemini.append(error_response)
                      continue

            logger.info(f"Preparing FC execution: {function_name}({args})")

            if function_name in available_functions_map:
                handler_func = available_functions_map[function_name]
                fc_exec_tasks.append(
                    execute_function_call(handler_func, args, original_chat_id, original_user_id)
                )
            else:
                logger.error(f"{model_name_str} requested unknown function: '{function_name}'")
                async def unknown_func_error(name=function_name): return {"error": f"Function '{name}' is not defined or available."}
                fc_exec_tasks.append(unknown_func_error())

        if len(fc_exec_tasks) != len(function_calls_to_process):
            logger.critical(f"Mismatch in FC count ({len(function_calls_to_process)}) and tasks created ({len(fc_exec_tasks)}). Aborting step.")
            missing_count = len(function_calls_to_process) - len(fc_exec_tasks)
            for i in range(missing_count):
                 error_name = function_calls_to_process[len(fc_exec_tasks) + i].name if len(fc_exec_tasks) + i < len(function_calls_to_process) else f"unknown_error_placeholder_{i}"
                 error_part = Part(function_response=FunctionResponse(name=error_name, response={"error": "Internal error creating execution task."}))
                 response_parts_for_gemini.append(error_part)
            if not response_parts_for_gemini: break

        try:
             logger.debug(f"Executing {len(fc_exec_tasks)} function call tasks...")
             results = await asyncio.gather(*fc_exec_tasks, return_exceptions=True)
             logger.debug(f"Finished executing tasks. Got {len(results)} results.")
        except Exception as gather_err:
             logger.error(f"Error during asyncio.gather for FC execution: {gather_err}", exc_info=True)
             results = [{"error": f"Failed to gather FC results: {gather_err}"}] * len(function_calls_to_process)

        # --- ИЗМЕНЕНО: Обновляем имя последней УСПЕШНОЙ функции ЗДЕСЬ ---
        for i, result_or_exception in enumerate(results):
            fc_name = function_calls_to_process[i].name # Получаем имя соответствующего вызова
            # Проверяем, был ли результат ошибкой
            is_error = isinstance(result_or_exception, Exception) or \
                       (isinstance(result_or_exception, dict) and 'error' in result_or_exception)
            if not is_error:
                # Если НЕ ошибка, обновляем имя последней успешной функции
                last_successful_fc_name = fc_name
                logger.debug(f"Updated last_successful_fc_name to '{fc_name}'")
            else:
                 # Логируем ошибку, если она была
                 if isinstance(result_or_exception, Exception):
                      logger.error(f"FC task '{fc_name}' failed with exception: {result_or_exception}", exc_info=result_or_exception)
                 elif isinstance(result_or_exception, dict) and 'error' in result_or_exception:
                      logger.error(f"FC task '{fc_name}' returned error dict: {result_or_exception}")
        # ------------------------------------------------------------

        # --- Формируем FunctionResponse Parts на основе результатов (логика без изменений) ---
        if len(results) != len(function_calls_to_process):
             logger.critical(f"Result count ({len(results)}) mismatch with FC count ({len(function_calls_to_process)}) after gather. Aborting step.")
             for i in range(len(function_calls_to_process)):
                 if i >= len(results):
                      fc_name = function_calls_to_process[i].name
                      error_part = Part(function_response=FunctionResponse(name=fc_name, response={"error": "Missing result after execution."}))
                      response_parts_for_gemini.append(error_part)
                 elif i >= len(response_parts_for_gemini):
                      fc_name = function_calls_to_process[i].name
                      result_or_exception = results[i]
                      response_data = {}
                      if isinstance(result_or_exception, Exception):
                           response_data = {"error": f"Execution failed: {result_or_exception}"}
                      elif isinstance(result_or_exception, dict):
                           response_data = result_or_exception
                      else:
                           response_data = {"result": str(result_or_exception)}
                      try:
                           response_part = Part(function_response=FunctionResponse(name=fc_name, response=response_data))
                           response_parts_for_gemini.append(response_part)
                      except Exception as part_create_err:
                           logger.error(f"Failed to create FunctionResponse Part for '{fc_name}': {part_create_err}", exc_info=True)
                           error_part = Part(function_response=FunctionResponse(name=fc_name, response={"error": f"Internal error creating response part: {part_create_err}"}))
                           response_parts_for_gemini.append(error_part)
             if not response_parts_for_gemini: break
        else:
            for i, result_or_exception in enumerate(results):
                # Пропускаем, если часть уже была добавлена из-за ошибки ранее
                # (Эта проверка нужна, если в блоке if выше добавлялись error_part)
                if i < len(response_parts_for_gemini):
                    continue

                fc_name = function_calls_to_process[i].name
                response_data = {}  # Инициализируем response_data для каждого результата

                if isinstance(result_or_exception, Exception):
                    # Если результат - исключение
                    response_data = {"error": f"Execution failed: {result_or_exception}"}
                elif isinstance(result_or_exception, dict):
                    # Если результат уже словарь (например, от execute_terminal_command_in_env)
                    response_data = result_or_exception
                elif isinstance(result_or_exception, str):
                    # Если результат - строка (например, от read_file_from_env)
                    result_str = result_or_exception
                    # Простое экранирование обратных слешей и кавычек
                    processed_result_str = result_str.replace('\\', '\\\\').replace('"', '\\"')
                    # Убираем потенциально проблемный перенос строки в конце, если он есть
                    # Сначала проверяем экранированный перенос
                    if processed_result_str.endswith('\\n'):
                        processed_result_str = processed_result_str[:-2]
                    # Потом проверяем обычный (на всякий случай)
                    elif processed_result_str.endswith('\n'):
                        processed_result_str = processed_result_str[:-1]

                    response_data = {"result": processed_result_str}
                    logger.debug(
                        f"Processed string result for FunctionResponse '{fc_name}': {str(response_data)[:100]}...")  # Логируем обработанный результат
                else:
                    # Для других типов данных (int, float, bool, NoneType и т.д.)
                    # Просто преобразуем в строку и помещаем в "result"
                    # Преобразуем None в пустую строку или строку "None" по желанию
                    processed_result = "" if result_or_exception is None else str(result_or_exception)
                    response_data = {"result": processed_result}

                # Пытаемся создать Part с FunctionResponse
                try:
                    response_part = Part(function_response=FunctionResponse(name=fc_name, response=response_data))
                    response_parts_for_gemini.append(response_part)
                except Exception as part_create_err:
                    logger.error(
                        f"Failed to create FunctionResponse Part for '{fc_name}' with data {str(response_data)[:100]}...: {part_create_err}",
                        exc_info=True)
                    # Создаем ответ об ошибке, если не удалось создать основной
                    error_part = Part(function_response=FunctionResponse(name=fc_name, response={
                        "error": f"Internal error creating response part: {part_create_err}"}))
                    # Добавляем только если не было добавлено ранее (на случай дублирования ошибок)
                    if i >= len(response_parts_for_gemini):
                        response_parts_for_gemini.append(error_part)


        if response_parts_for_gemini:
             if len(response_parts_for_gemini) != len(function_calls_to_process):
                  logger.critical(f"FATAL MISMATCH before sending to API: Expected {len(function_calls_to_process)} parts, got {len(response_parts_for_gemini)}. ABORTING API CALL.")
                  current_response = None
                  break

             logger.info(f"Sending {len(response_parts_for_gemini)} function responses back to {model_name_str}...")
             try:
                 loop = asyncio.get_running_loop()
                 next_response = await loop.run_in_executor(
                     None, gemini_api.send_message_to_gemini, model_instance, chat_session, response_parts_for_gemini
                 )

                 if next_response is None:
                      logger.error(f"send_message_to_gemini returned None after sending FC responses for {model_name_str}. Ending FC cycle.")
                      current_response = None
                      break

                 if chat_session.history and isinstance(chat_session.history[-1], Content) and chat_session.history[-1].role == 'model':
                      last_content = chat_session.history[-1]
                      current_response = MockResponse(last_content)
                      logger.debug("Updated current_response from chat history for next FC step.")
                 else:
                      logger.warning("Chat history not updated correctly or last entry not from model after sending FC responses. Ending FC cycle.")
                      current_response = None
                      break

             except Exception as send_back_err:
                 logger.error(f"Exception sending function responses back to Gemini: {send_back_err}", exc_info=True)
                 current_response = None
                 break
        else:
            logger.warning("No function response parts were generated, though FCs were detected. Ending FC cycle.")
            break

    if step == max_steps:
        logger.warning(f"Max FC steps ({max_steps}) reached for {model_name_str} in chat {original_chat_id}.")

    logger.info(f"--- FC Cycle Finished for {model_name_str} (Chat: {original_chat_id}) ---")

    # --- ИЗМЕНЕНО: Возвращаем историю и имя последней УСПЕШНОЙ функции ---
    return chat_session.history, last_successful_fc_name
    # ----------------------------------------------------------------