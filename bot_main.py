# bot_main.py
import asyncio
import logging

# --- Локальные импорты ---
from bot_loader import dp, bot # Импорт Dispatcher и Bot
from bot_lifecycle import on_startup, on_shutdown # Функции жизненного цикла
from handlers import private, group # Важно импортировать, чтобы хендлеры зарегистрировались

# Настройка логирования выполняется в bot_config при импорте
logger = logging.getLogger(__name__)

async def main():
    """Основная функция запуска бота."""
    # Регистрация функций жизненного цикла
    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    # Удаление вебхука и пропуск старых обновлений
    await bot.delete_webhook(drop_pending_updates=True)

    logger.info("Starting bot polling...")
    try:
        # Запуск поллинга
        # Хендлеры из private.py и group.py уже зарегистрированы благодаря импорту выше
        await dp.start_polling(bot)
    finally:
        # Корректное завершение сессии бота при остановке поллинга
        logger.info("Stopping polling. Closing bot session...")
        await bot.session.close()
        logger.info("Bot session closed.")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped by user!")
    except Exception as e:
        logger.critical(f"Unhandled exception at main level: {e}", exc_info=True)
        # Завершение с ошибкой
        exit(1)