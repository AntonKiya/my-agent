---
name: weather-forecast
description: "Answer weather questions with the built-in weather forecast tool. TRIGGER when: the user asks about current weather, today's weather, tomorrow's weather, rain, snow, temperature, wind, what to wear because of weather, or a forecast for the next few days/week."
---

# Weather Forecast Skill

Answer practical weather questions with the `get_weather_forecast` tool.

## Workflow

1. Determine the location from the current message or clear recent conversation context.
2. Determine the forecast period from the user's wording.
3. Call `get_weather_forecast` once with location, period, and location_language.
4. If the tool returns `needs_location`, `location_not_found`, or `ambiguous_location`, ask one short clarification question.
5. Answer compactly from the tool result.

## Tool

Use:
- `get_weather_forecast(location, period, location_language)` — resolves the place and returns current/daily/hourly forecast data.

The model should pass the location as normal human text from the user or context. Do not translate or canonicalize city names yourself. The tool handles URL encoding, geocoding, forecast coordinates, and ambiguous matches.

## Location Policy

Use a location from recent context when it is clearly the intended place:
- "I live in Kazan" followed later by "what's the weather tomorrow?" means Kazan.
- "I'm going to Saint Petersburg tomorrow" followed by "will it rain there?" means Saint Petersburg.
- "I'm from Moscow and flying to Sochi. What's the weather?" means Sochi.

Ask for the city/place when:
- no location is named and no clear context location exists;
- several places are mentioned and the intended one is unclear;
- the tool returns `ambiguous_location` or `location_not_found`.

Ask only one short question, e.g. "Для какого города посмотреть погоду?"

## Period Policy

Infer the period from meaning:
- "сейчас", "на улице", "какая погода" when the user wants the current state: `now`
- "сегодня", "вечером", "что надеть": `today`
- "завтра": `tomorrow`
- "на неделю", "в ближайшие дни", "прогноз": `week`

If the user asks for a forecast and the period is not clear, use `week`.

## Location Language Policy

Pass the language code that matches the user's language or the location spelling:
- Russian conversation or Cyrillic location: `ru`
- English conversation or Latin location: `en`

Do not translate the location just to fit the language.

## Response Policy

Be concise and practical:
- mention temperature, apparent temperature when useful, precipitation/rain/snow risk, wind, and notable daily changes;
- for `week`, summarize the trend and call out the wettest/coldest/warmest days if visible;
- do not expose raw coordinates, API parameters, or weather codes;
- if the result is uncertain or the tool asks for clarification, ask the user instead of guessing.
