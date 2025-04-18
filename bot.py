import logging
import re
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from decouple import config
from transformers import pipeline
from deep_translator import GoogleTranslator

# Налаштування логування
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Завантаження токена
TELEGRAM_TOKEN = config("TELEGRAM_TOKEN", default=None)
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN не знайдено в .env")
    exit(1)
logger.info("TELEGRAM_TOKEN успішно завантажено")

# Ініціалізація GPT-2 і перекладача
try:
    logger.info("Ініціалізація моделі distilgpt2...")
    generator = pipeline("text-generation", model="distilgpt2", device=-1, model_kwargs={"cache_dir": "./model_cache"})
    logger.info("Модель distilgpt2 ініціалізовано")
    translator = GoogleTranslator(source="auto", target="en")
    logger.info("Перекладач ініціалізовано")
except Exception as e:
    logger.error(f"Помилка ініціалізації GPT-2 або перекладача: {e}")
    exit(1)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Вітає користувача при запуску бота."""
    try:
        keyboard = [
            [InlineKeyboardButton("Дізнатися більше", callback_data="info")],
            [InlineKeyboardButton("Задати питання", callback_data="ask")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Привіт! Я AI-бот із GPT-2. Вибери опцію!", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Помилка в команді /start: {e}")
        await update.message.reply_text("Вибач, щось пішло не так. Спробуй ще раз!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показує довідку про бота."""
    try:
        await update.message.reply_text(
            "Я AI-бот із GPT-2! Використовуй:\n"
            "/start - Почати\n"
            "/help - Допомога\n"
            "/lang - Вибрати мову\n"
            "/stats - Показати статистику\n"
            "Просто напиши текст, і я згенерую відповідь!"
        )
    except Exception as e:
        logger.error(f"Помилка в команді /help: {e}")
        await update.message.reply_text("Вибач, щось пішло не так. Спробуй ще раз!")

async def lang_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Дозволяє вибрати мову відповідей."""
    try:
        keyboard = [
            [InlineKeyboardButton("Українська", callback_data="lang_uk")],
            [InlineKeyboardButton("English", callback_data="lang_en")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Вибери мову:", reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Помилка в команді /lang: {e}")
        await update.message.reply_text("Вибач, щось пішло не так.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показує статистику запитів."""
    try:
        query_count = context.bot_data.get("query_count", 0)
        await update.message.reply_text(f"Бот обробив {query_count} запитів!")
    except Exception as e:
        logger.error(f"Помилка в команді /stats: {e}")
        await update.message.reply_text("Вибач, щось пішло не так.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обробляє натискання кнопок."""
    try:
        query = update.callback_query
        await query.answer()
        if query.data == "info":
            await query.message.reply_text("Я використовую DistilGPT-2 для генерації тексту!")
        elif query.data == "ask":
            await query.message.reply_text("Напиши своє питання, і я відповім!")
        elif query.data.startswith("lang_"):
            lang = query.data.split("_")[1]
            context.user_data["lang"] = lang
            await query.message.reply_text(f"Мову змінено на {lang}!")
    except Exception as e:
        logger.error(f"Помилка в обробці кнопки: {e}")
        await query.message.reply_text("Вибач, щось пішло не так.")

async def ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Генерує відповідь за допомогою GPT-2."""
    try:
        user_input = update.message.text
        if len(user_input) > 500:
            await update.message.reply_text("Запит занадто довгий! Спробуй до 500 символів.")
            return
        lang = context.user_data.get("lang", "uk")
        context.bot_data["query_count"] = context.bot_data.get("query_count", 0) + 1
        logger.info(f"Отримано запит: {user_input} (мова: {lang})")
        try:
            translated_input = translator.translate(user_input)
            logger.info(f"Перекладено на англійську: {translated_input}")
            prompt = f"Question: {translated_input}\nAnswer in a clear and concise way: "
        except Exception as e:
            logger.warning(f"Помилка перекладу: {e}. Використовую оригінальний запит.")
            prompt = f"Question: {user_input}\nAnswer in a clear and concise way: "
        response = generator(
            prompt,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            truncation=True,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=generator.tokenizer.eos_token_id
        )[0]["generated_text"]
        logger.info(f"Сира відповідь моделі: {response}")
        cleaned_response = response.replace(prompt, "").strip()
        # Фільтруємо обірвані речення
        cleaned_response = re.sub(r'\.\s*$|\s*$', '.', cleaned_response)
        if not cleaned_response or len(cleaned_response) < 20 or len(cleaned_response.split()) < 5:
            cleaned_response = "I couldn't generate a good answer. Try again!" if lang == "en" else "Не вдалося згенерувати хорошу відповідь. Спробуй ще!"
        logger.info(f"Очищена відповідь: {cleaned_response}")
        try:
            translated_response = GoogleTranslator(source="en", target=lang).translate(cleaned_response)
            logger.info(f"Перекладено на {lang}: {translated_response}")
        except Exception as e:
            logger.warning(f"Помилка зворотного перекладу: {e}. Повертаю англійську відповідь.")
            translated_response = cleaned_response
        await update.message.reply_text(translated_response)
    except Exception as e:
        logger.error(f"Помилка в генерації відповіді: {e}")
        await update.message.reply_text("Ой, не можу згенерувати відповідь. Спробуй ще раз!")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обробляє помилки бота."""
    logger.error(f"Помилка: {context.error}")
    if update and update.message:
        await update.message.reply_text("Вибач, сталася помилка. Спробуй пізніше!")

def main():
    """Запускає бота."""
    try:
        logger.info("Ініціалізація Telegram Application...")
        app = Application.builder().token(TELEGRAM_TOKEN).read_timeout(10).write_timeout(10).build()
        logger.info("Application ініціалізовано")
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("lang", lang_command))
        app.add_handler(CommandHandler("stats", stats_command))
        app.add_handler(CallbackQueryHandler(button_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai_reply))
        app.add_error_handler(error_handler)
        logger.info("Хендлери додано. Починаємо polling...")
        app.run_polling()
        logger.info("Polling завершено")
    except Exception as e:
        logger.error(f"Помилка запуску бота: {e}")
        raise

if __name__ == "__main__":
    main()