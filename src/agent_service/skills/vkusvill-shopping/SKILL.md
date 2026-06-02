---
name: vkusvill-shopping
description: "Assemble a grocery cart and return a VkusVill (grocery store) cart link from a natural-language food or shopping request. TRIGGER when: the user wants to buy food, order groceries, cook something, eat at home, or get products — even if VkusVill is not mentioned by name (it is the only available store)."
---

# VkusVill Shopping Skill

Turn a free-form food or grocery request into a VkusVill (grocery store) cart link with minimal friction.

## Workflow

1. Parse the request into a list of items with their attributes (type, quantity, required/excluded descriptors).
2. If the request is materially ambiguous, ask one short clarification round (max 2–3 questions).
3. For each item: search, select the best match, use analogs if needed.
4. Build one cart link from all selected products.
5. Return the link with a brief summary.

## Tools

**Use:**
- `mcp_vkusvill_vkusvill_products_search` — primary discovery for every item
- `mcp_vkusvill_vkusvill_product_details` — only when search results are insufficient to select confidently (dietary constraints, 2–3 very close candidates)
- `mcp_vkusvill_vkusvill_product_analogs` — fallback when no good match is found in search
- `mcp_vkusvill_vkusvill_cart_link_create` — final step only

**Never use for shopping tasks:**
- `web_search`, `web_fetch`, `exec` — forbidden, without exceptions

## Clarification Policy

Ask only when ambiguity would materially change product selection:
- Category is too broad with no disambiguating detail (e.g. "wine", "meat", "pizza")
- A missing attribute strongly affects the result (red vs white, cooked vs raw, sweet vs dry)
- Multiple incompatible product types match equally well

Do not ask when:
- The user gave enough detail for a practical choice
- Remaining uncertainty is minor or a sensible default is obvious
- The user is intentionally leaving the choice to the agent (e.g. "something sweet and baked" — just pick a cookie or muffin)

**Limits:** one clarification round, max 2–3 short questions in a single message. Do not follow up with a second clarification round unless the user introduces new ambiguity in their reply.

## Search Policy

For each item:
1. Build a concise, specific query from the user's own wording.
2. Run `products_search`. Review the full first page of results — do not just take the first item.
3. If no result on the first page is a good match, run **one** rephrased query (synonym, different term order). No further retries after that.
4. Never run more than 2 queries per item. Never paginate beyond the first page.

Sort order:
- Default: `popularity`
- User asks for best or highest-rated: `rating`
- User asks for cheaper: `price_asc`
- User explicitly wants new arrivals: `new`

## Candidate Selection

Pick the best match from results — not the first one, not the highest-rated one by default.

Rank candidates in this order:
1. Hard exclusions from the user must be absent
2. Correct product category
3. Required descriptors present (size, format, origin, style)
4. Soft preference hints (treat as ranking signal, not hard filter)
5. Lexical and semantic closeness to the user's wording
6. Rating and popularity as a tiebreaker only

If 2–3 candidates are very close and the difference matters, call `product_details` on them to break the tie. Do not call `product_details` on every item by default.

## Fallback

If no confident match after 2 queries:
1. Try `product_analogs` on the closest result found.
2. If analogs also yield no good match, briefly tell the user and suggest the nearest available substitute.

Do not silently add a wrong product. Do not fabricate confidence.

## Cart Building

- Collect `xml_id` and quantity `q` (number of units) for each selected product.
- `q` must be between `0.01` and `40`.
- One `cart_link_create` call supports 1–20 products.
- If the list exceeds 20 items: ask the user to narrow it down. Do not silently drop items.
- Default quantity is `1` when the user did not specify.

## Response Format

**On clarification:**
One short message with 2–3 concrete questions only.

**On success:**
Cart link on the first line, then a compact list of what was added. Note any substitutions.

**On partial failure:**
Return the link for items that succeeded. Clearly state which items could not be matched and why.

## Error Handling

- Temporary tool failure: retry once with a simpler call; if still failing, tell the user the VkusVill service is temporarily unavailable.
- No match found: report honestly and offer the nearest available alternative.
- Cart creation fails after successful selection: tell the user product selection succeeded but the link could not be created; invite them to retry.
