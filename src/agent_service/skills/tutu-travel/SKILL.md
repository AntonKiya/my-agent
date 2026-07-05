---
name: tutu-travel
description: "Use Tutu.ru MCP for tickets, routes, flights, trains, buses, commuter trains, hotels/lodging, apartments, studios, apart-hotels, guesthouses, transport comparisons, checkout links, and follow-up requests that select, book, or ask for a link to a previously shown Tutu option. Use this instead of web tools unless the user explicitly asks for internet, official sites, comparison with other platforms, or non-Tutu sources."
---

# Tutu Travel Skill

Use Tutu MCP as the only source for Tutu tickets, lodging, prices, availability, schedules, details, checkout links, and changes to dates, guests, passengers, seats, rooms, or fares. The user completes checkout on Tutu.

## Workflow

1. Classify the request: hotel, specific transport, transport-neutral route, or transport plus lodging.
2. If required search fields are missing, ask one short clarification.
3. If required search fields are present, call the relevant Tutu search tool.
4. Return a compact decision-focused comparison.
5. Call details tools only for selected or shortlisted options.
6. Create a checkout link only after a concrete user choice.

## Tools

Use only these Tutu MCP tools for Tutu tasks:

- `mcp_tutu_search_hotels`
- `mcp_tutu_search_avia`
- `mcp_tutu_search_rail`
- `mcp_tutu_search_bus`
- `mcp_tutu_search_etrain`
- `mcp_tutu_search_multitransport`
- `mcp_tutu_get_offer_details`
- `mcp_tutu_get_rail_seatmap`
- `mcp_tutu_create_checkout_link`

Do not use web search, web research, or web page reading for Tutu tickets, lodging, prices, schedules, availability, offer details, checkout links, or Tutu link validation.

Use external websites only when the user explicitly asks for internet search, official sites, comparison with other booking platforms, or non-Tutu sources.

If a Tutu link looks broken, the user changes dates/guests/passengers/seats/rooms/fares, or the user asks for another link, rerun the relevant Tutu search if needed and call `mcp_tutu_create_checkout_link` for the selected option.

Never edit, reconstruct, validate, or replace Tutu links through external websites.

## Trigger Policy

Use this skill for:

- tickets, flights, trains, buses, commuter trains;
- hotels, lodging, apartments, studios, apart-hotels, guesthouses, non-hotel stays;
- transport comparisons and "how do I get there" requests;
- follow-up selections from the latest Tutu list, including "first", "second", "cheapest", a hotel/transport name, "book it", "reserve it", or "give me the link".

Do not use this skill for visas, entry rules, insurance, sightseeing, tours, restaurants, walking routes, car rental, or general travel advice unless the user asks for tickets or lodging.

## Required Fields

Ask at most one short clarification. Ask only for missing required fields.

Transport requires:

- origin;
- destination;
- travel date;
- passenger count when the user mentions a group, children, or family without specifying the party.

Hotels require:

- city or location;
- check-in date;
- check-out date;
- guest count;
- child ages when children are mentioned.

Do not ask for optional preferences before the first search: budget, hotel area, train class, baggage, airline, station/airport, breakfast, cancellation, or transport type for "how do I get there".

Resolve relative dates before tool calls. Convert "tomorrow", "Friday", "this weekend", and "next week" to exact `YYYY-MM-DD` dates. State the searched exact dates in the answer.

If any resolved travel, check-in, or check-out date is in the past, do not call Tutu. Ask for future dates.

If origin, destination, and date are present for a transport-neutral route request, call `mcp_tutu_search_multitransport`. Do not ask for transport preference.

## Tool Choice

- No transport specified; route, cheapest, fastest, or all options request -> `mcp_tutu_search_multitransport`.
- Flight, plane, airfare, airline, route, airport-oriented request -> `mcp_tutu_search_avia`.
- Train, railway, RZD, compartment, platzkart, sleeper, berth, lower/upper seat -> `mcp_tutu_search_rail`.
- Bus, coach, bus station -> `mcp_tutu_search_bus`.
- Commuter train, suburban train, elektrichka -> `mcp_tutu_search_etrain`.
- Hotel, lodging, room, apartment, studio, apart-hotel, guesthouse, non-hotel stay -> `mcp_tutu_search_hotels`.

Do not call individual transport search tools when `mcp_tutu_search_multitransport` fits.

For multitransport, summarize cheapest, fastest, and balanced choices. Do not dump raw rows.

## Search Defaults

Use compact search results first. Use full/detail views only for rules, room data, fare data, reviews, seats, baggage, amenities, or checkout-critical details.

First-pass result sizes:

- transport: show 3 options; show 4-5 only when options differ by mode, price, time, directness, or user request;
- hotels: search 8-10 results and show 5-8;
- transport plus lodging: show 2-3 combinations.

Ranking:

- cheapest request -> price;
- fastest request -> duration/time;
- morning/evening request -> departure time;
- hotels -> stated preferences, then location, rating, cancellation/breakfast, and total stay price when calculable.

For hotel prices, label price as per-night or full-stay. If the tool returns per-night price and nights are known, calculate the stay total.

## Details And Seatmaps

Call `mcp_tutu_get_offer_details` only when details affect selection or checkout: baggage, refund/exchange, cancellation, breakfast, payment type, room type, beds, amenities, fare rules, or train/bus/hotel details.

Fetch details for one selected option or 2-3 shortlisted options. Do not fetch details for every result.

Call `mcp_tutu_get_rail_seatmap` only for train seat questions: lower/upper, nearby, side/non-side, car number, farther from WC, gendered compartment, or exact seats.

Do not show raw full seatmaps. Summarize relevant seats and conditions.

## Checkout Link Policy

A concrete choice includes a selected number, "cheapest direct", a specific transport option, a hotel name, a room/rate, or a follow-up selection from the latest Tutu list.

Use the selected Tutu result's `selection_id` to create checkout links.

Call `mcp_tutu_create_checkout_link` with `selection_id` after a concrete user choice. Do not build checkout links from search URLs or manually supplied fields.

Never invent, template, edit, shorten, reconstruct, or reuse a checkout link from memory.

For avia, if `mcp_tutu_create_checkout_link` returns `kind="search_redirect"`, say Tutu returned only a search page, not a direct booking link. Do not present `search_results_url` as checkout.

For hotels, a hotel-page deeplink with dates and guests is valid after the user selects a hotel. Create a room/rate checkout link only after the user selects a specific room/rate and `offer_pack_hash` is available from `mcp_tutu_get_offer_details`.

If the user changes the option, party, dates, seats, fare, room/rate, or hotel, call `mcp_tutu_create_checkout_link` again and use the newly returned URL.

If link creation fails, say that the link could not be created. Do not provide a manual fallback URL.

Checkout response:

```text
Tutu checkout link: {checkout_url}

Selected: {short option description}.
```

## Response Format

Use compact Markdown tables only when comparing multiple options by price, time, duration, rating, terms, or location.

Do not use a table for one selected option or one link.

Transport rows must include price, departure/arrival, duration, carrier/mode, route/stations/airports when returned, and a short label: Cheapest, Fastest, Balanced, Direct, Morning, or Overnight.

Hotel rows must include hotel, location/address, rating, total price when calculable, breakfast/cancellation when returned, and one short reason.

Ask a narrowing question only when several shown options still match the user's stated goal equally.

## Transport Plus Lodging

For trips with transport plus lodging, search transport with `mcp_tutu_search_multitransport` unless a transport type is specified, and search hotels with `mcp_tutu_search_hotels`.

Do not fetch details for all results. Build 2-3 combinations and create separate checkout links for selected components.

Do not present transport plus hotel as one package unless Tutu returns a single package product.

## Selection From Context

Resolve "first", "second", "cheapest", option numbers, hotel names, transport names, and similar references against the latest clear Tutu list.

If several lists exist and the reference is ambiguous, ask which list or component the user means.

## Accuracy

Treat Tutu MCP as the source of truth for prices, availability, schedules, seats, rooms, fares, terms, and links.

Use only fields returned by Tutu MCP. If baggage, cancellation, bed type, amenities, seat availability, rating, or another requested field is absent, say it is not specified in the Tutu result.

Do not relax hard constraints without user consent: direct only, specific transport only, budget cap, free cancellation only, lower berth only.

## Errors

If a Tutu tool returns `error.user_message`, show that message and ask for the smallest useful correction. Do not reinterpret it as an upstream outage.

If a Tutu tool returns an error result, do not retry the same parameters.

If no results are returned, say Tutu did not return suitable options and suggest one next step: nearby date, relaxed filter, another transport type, another area, or another budget.

Do not use web tools as an error fallback for Tutu tasks.
