---
name: tutu-travel
description: "Plan trips through Tutu.ru: find flights, trains, buses, commuter trains, hotels/lodging including apartments, studios, private studios, apart-hotels, guesthouses, and non-hotel stays, compare ways to get from one place to another, and create checkout links. TRIGGER when: the user wants tickets, lodging, travel options, a transport comparison, or trip planning. Prefer this skill over web search unless the user explicitly asks for internet or external sources."
---

# Tutu Travel Skill

Find Tutu tickets, transport options, hotels, and trips with transport plus lodging. Search first when required inputs are present; clarify only when a search would be impossible or materially poor.

Tutu MCP returns live Tutu options and checkout links. The user completes checkout on Tutu.

## Workflow

1. Classify the request: hotel, specific transport, transport-neutral route, or transport plus lodging.
2. Ask one short clarification only if required search data is missing.
3. Search Tutu immediately when required inputs are present.
4. Return a compact comparison focused on the decision.
5. Fetch details only for selected or shortlisted options.
6. Create a checkout link only after a concrete user choice.

## Tools

Use these Tutu MCP tools:

- `mcp_tutu_search_hotels`
- `mcp_tutu_search_avia`
- `mcp_tutu_search_rail`
- `mcp_tutu_search_bus`
- `mcp_tutu_search_etrain`
- `mcp_tutu_search_multitransport`
- `mcp_tutu_get_offer_details`
- `mcp_tutu_get_rail_seatmap`
- `mcp_tutu_create_checkout_link`

Use Tutu MCP as the only source for tickets, hotels, prices, availability, schedules, offer details, checkout links, and changes to dates, guests, passengers, seats, rooms, or fares.

Do not call web search, web research, or web page reading tools for these tasks. If a Tutu link looks broken, the user changes dates/guests/passengers, or the user asks for another link, call the relevant Tutu search tool again and then `mcp_tutu_create_checkout_link` with the selected offer's `checkout_ref`.

Never edit, reconstruct, validate, or replace Tutu links via external websites. Use external websites only when the user explicitly asks to search the web, check an official site, compare with other booking platforms, or use non-Tutu sources.

## Trigger Policy

Use this skill for tickets, flights, trains, buses, commuter trains, hotels/lodging, apartments, studios, private studios, apart-hotels, guesthouses, non-hotel stays, transport comparison, and requests to get from one city, station, or airport to another.

Do not use this skill for visas, entry rules, insurance, sightseeing, tours, restaurants, walking routes, car rental, or general travel advice unless the user asks for tickets or lodging.

## Clarification Policy

Ask at most one short clarification round. Ask only for missing required data.

Transport requires:

- origin;
- destination;
- travel date;
- passenger count only when the user mentions a group, children, or family but does not specify the party.

Hotels require:

- city or location;
- check-in date;
- check-out date;
- guest count;
- child ages if children are mentioned.

Do not ask upfront for optional preferences: budget, hotel area, train class, baggage, airline, station/airport, or transport type for a "how do I get there" request.

Resolve relative dates before tool calls. "Tomorrow", "Friday", "this weekend", and "next week" must become exact `YYYY-MM-DD` dates in tool arguments. State the exact dates searched in the answer.

If a resolved travel date, check-in date, or check-out date is in the past, do not call Tutu. Ask which future year or dates the user means.

If the user asks how to get from one place to another and gives origin, destination, and date, do not ask about transport preference. Use `mcp_tutu_search_multitransport`.

## Tool Choice

- No transport specified; user asks how to get there, cheapest, fastest, or all options -> `mcp_tutu_search_multitransport`.
- Flight, plane, airfare, airline, route, or airport-oriented flight request -> `mcp_tutu_search_avia`.
- Train, railway, RZD, compartment, platzkart, sleeper, berth, lower/upper seat -> `mcp_tutu_search_rail`.
- Bus, coach, or bus station -> `mcp_tutu_search_bus`.
- Commuter train, suburban train, or elektrichka -> `mcp_tutu_search_etrain`.
- Hotel, lodging, room, place to stay, apartment, studio, apart-hotel, guesthouse, private studio, or non-hotel stay -> `mcp_tutu_search_hotels`.

Do not run all individual transport tools when `mcp_tutu_search_multitransport` fits.

For multitransport, summarize cheapest, fastest, and balanced choices instead of dumping raw rows.

For hotels, show more options than transport when useful.

## Search Defaults

Use compact search results first. Use full/detail views only when the user needs specific rules, room data, fare data, reviews, or seat information.

Default first-pass result sizes:

- transport: 3-5 useful options;
- hotels: 8-10 search results, usually show 5-8;
- transport plus lodging: 2-3 combinations.

Ranking:

- cheapest requests -> price;
- fastest requests -> duration/time;
- morning/evening requests -> departure time;
- hotels -> match preferences first, then location, rating, cancellation/breakfast, and total stay price when calculable.

For hotel prices, be explicit whether the price is per night or for the full stay. If the tool returns a per-night price and nights are known, calculate the stay total.

## Details And Seatmaps

Use `mcp_tutu_get_offer_details` only when details affect selection or checkout: baggage, refund/exchange, cancellation, breakfast, payment type, room type, beds, amenities, fare rules, or train/bus/hotel details.

Fetch details for one selected option or 2-3 shortlisted options, not every result.

Use `mcp_tutu_get_rail_seatmap` only for train seat questions: lower/upper, nearby, side/non-side, car number, farther from WC, gendered compartment, or exact seats.

Do not show raw full seatmaps. Summarize relevant seats and conditions.

## Checkout Link Policy

Create checkout links only after a concrete choice: selected number, unambiguous "cheapest direct", specific transport option, or specific hotel room/rate when required.

Always call `mcp_tutu_create_checkout_link` with the selected offer's `checkout_ref`.

Never invent, template, edit, shorten, reconstruct, or reuse a checkout link from memory.

If the user changes the option, party, dates, seats, fare, room/rate, or hotel, call `mcp_tutu_create_checkout_link` again and use the newly returned URL.

If link creation fails, say that the link could not be created. Do not provide a manual fallback URL.

## Response Format

Use compact Markdown tables when comparing multiple options by price, time, duration, rating, terms, or location. Do not use a table for one selected option or one link.

Transport: usually show 3 best options; show 4-5 only when options are meaningfully different or the user asks for more. Use labels such as Cheapest, Fastest, Balanced, Direct, Morning, Overnight.

Hotels: usually show 5-8 useful options. Include hotel, location/address, rating, total price when calculable, breakfast/cancellation if returned, and why it fits. Invite the user to narrow by area, budget, rating, breakfast, cancellation, or room type when there are many plausible hotels.

Transport plus lodging: show 2-3 combinations, such as cheaper, more convenient, and balanced.

Checkout response:

```text
Tutu checkout link: {checkout_url}

Selected: {short option description}.
```

## Transport Plus Lodging

For trips with transport plus lodging, resolve dates, search transport with `mcp_tutu_search_multitransport` unless a transport type is specified, and search hotels with `mcp_tutu_search_hotels`.

Do not fetch details for all results. Build 2-3 clear combinations and create separate checkout links for selected components.

Do not present transport plus hotel as one package unless Tutu returns a single package product.

## Selection From Context

Resolve "first", "second", "cheapest", and similar references against the latest clear list. If there are several lists, ask which component or list the user means.

## Accuracy

Treat Tutu MCP as the source of truth for prices, availability, schedules, seats, rooms, fares, terms, and links.

Never invent missing data. If baggage, cancellation, bed type, amenities, seat availability, rating, or another requested field is absent, say it is not specified in the Tutu result.

## Errors

If no results are returned, say Tutu did not return suitable options and suggest one next step: nearby date, relaxed filter, another transport type, another area, or another budget.

If a Tutu tool returns an error result, do not retry the same parameters. Explain that Tutu could not run that search and ask for the smallest useful correction, usually future dates, guest/passenger count, another location, or a relaxed hard filter.

Do not relax hard constraints without user consent: direct only, specific transport only, budget cap, free cancellation only, lower berth only.
