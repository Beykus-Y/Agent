# bot_loader.py
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

from bot_config import BOT_TOKEN, logger

# Инициализация хранилища FSM (если потребуется в будущем)
storage = MemoryStorage()

# Инициализация бота с настройками по умолчанию
# Устанавливаем MarkdownV2, так как будем использовать экранирование
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN_V2)
)

# Инициализация диспетчера
dp = Dispatcher(storage=storage)

logger.info("Bot and Dispatcher initialized.")