---
name: reminders
description: "Create, list, and cancel user reminders. TRIGGER when: the user asks to be reminded once, repeatedly, daily, weekly, on certain days, during a time window, or asks to show/cancel reminders."
---

Use the reminder tools for reminders and recurring notifications.

## Rules

1. Create reminders with `create_once_reminder`, `create_weekly_reminder`, or `create_interval_window_reminder`; do not say you will remember something unless the tool succeeds.
2. Reminders are supported only in private conversations. If the tool reports unsupported conversation type, say reminders are only available in private chat for now.
3. Pass IANA timezones such as `Europe/Moscow`, `Europe/Sofia`, `Asia/Yerevan` to reminder tools.
4. For relative or calendar reminders such as "через пару минут", "через час", "завтра в 6", or "tomorrow evening", call `get_current_time` before computing local datetimes. Do not ask the user what time it is now.
5. If the user's profile timezone is available in tool context, the tools can use it. If timezone is unknown and not clear from the conversation, ask for the user's city in plain language before creating: "В каком ты городе? Мне нужно это, чтобы поставить напоминание по твоему местному времени."
6. If the user gives a city and the timezone is clear, pass the matching IANA timezone to `get_current_time` and reminder tools.
7. If frequency or time is missing entirely, ask one short clarification question.
8. For simple clear requests, create immediately and then confirm exact time and timezone.
9. If you used a vague-time default, include it in `assumptions` and mention it in the confirmation.
10. For requests with different messages on different days, create separate reminders. Use at most three reminders from one user message.
11. Keep medication names, dosages, money amounts, dates, and deadlines exact. Do not rewrite them creatively.
12. Treat "a couple of minutes" / "пара минут" as 2 minutes unless the user says otherwise.

## Defaults

- morning: 10:00
- middle of day / daytime: 13:00
- after lunch: 14:00
- afternoon / second half of the day: 16:00
- evening: 18:00
- daytime window: 09:00-18:00

## Supported Schedules

- `create_once_reminder`: one local datetime.
- `create_weekly_reminder`: days of week and one or more local times.
- `create_interval_window_reminder`: every N minutes within a local time window.
- `get_current_time`: current UTC time and, when timezone is known, current local time.

After successful creation, tell the user the exact schedule, timezone, and the nearest upcoming times returned by the tool.
