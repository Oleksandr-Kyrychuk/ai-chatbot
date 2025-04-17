from telegram.ext import Application, CommandHandler, MessageHandler, filters
from decouple import config
from transformers import pipeline

TELEGRAM_TOKEN = config("TELEGRAM_TOKEN")
generator = pipeline("text-generation", model="gpt2")

async def start(update, context):
    await update.message.reply_text("Привіт! Я AI-бот із GPT-2. Задай питання!")

async def ai_reply(update, context):
    user_input = update.message.text
    response = generator(user_input, max_length=50, num_return_sequences=1)[0]["generated_text"]
    await update.message.reply_text(response)

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, ai_reply))
    app.run_polling()

if __name__ == "__main__":
    main()