TELEGRAM_ONBOARDING_ANALYZE_IMAGE_CALLBACK_DATA = "onboarding:analyze_image"
TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA = "onboarding:lecture_summary"
TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA = "onboarding:generate_image"
TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA = "onboarding:web_research"
TELEGRAM_ONBOARDING_COMPARISON_CALLBACK_DATA = "onboarding:comparison"

TELEGRAM_ONBOARDING_ANALYZE_IMAGE_TEXT = "Мне нужно разобрать изображение"
TELEGRAM_ONBOARDING_LECTURE_SUMMARY_TEXT = "Мне нужно сделать конспект из файла"
TELEGRAM_ONBOARDING_GENERATE_IMAGE_TEXT = "Мне нужно создать изображение"
TELEGRAM_ONBOARDING_WEB_RESEARCH_TEXT = "Мне нужно найти информацию в интернете"
TELEGRAM_ONBOARDING_COMPARISON_TEXT = "Мне нужно сделать сравнение"

TELEGRAM_ONBOARDING_CALLBACK_TEXTS = {
    TELEGRAM_ONBOARDING_ANALYZE_IMAGE_CALLBACK_DATA: TELEGRAM_ONBOARDING_ANALYZE_IMAGE_TEXT,
    TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA: TELEGRAM_ONBOARDING_LECTURE_SUMMARY_TEXT,
    TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA: TELEGRAM_ONBOARDING_GENERATE_IMAGE_TEXT,
    TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA: TELEGRAM_ONBOARDING_WEB_RESEARCH_TEXT,
    TELEGRAM_ONBOARDING_COMPARISON_CALLBACK_DATA: TELEGRAM_ONBOARDING_COMPARISON_TEXT,
}
TELEGRAM_START_REPLY_MARKUP = {
    "inline_keyboard": [
        [
            {
                "text": "📷 Разбери фото конспекта",
                "callback_data": TELEGRAM_ONBOARDING_ANALYZE_IMAGE_CALLBACK_DATA,
            },
        ],
        [
            {
                "text": "📚 Вытащи главное из лекции",
                "callback_data": TELEGRAM_ONBOARDING_LECTURE_SUMMARY_CALLBACK_DATA,
            },
        ],
        [
            {
                "text": "🩻 Создай иллюстрацию",
                "callback_data": TELEGRAM_ONBOARDING_GENERATE_IMAGE_CALLBACK_DATA,
            },
        ],
        [
            {
                "text": "🔎 Найди материалы по теме",
                "callback_data": TELEGRAM_ONBOARDING_WEB_RESEARCH_CALLBACK_DATA,
            },
        ],
        [
            {
                "text": "📊 Проведи сравнение",
                "callback_data": TELEGRAM_ONBOARDING_COMPARISON_CALLBACK_DATA,
            },
        ],
    ],
}
