---
name: image-analysis
description: "Answer questions about images attached by the user. TRIGGER when: the current message contains attached image media_id markers and the user asks what is shown, asks to read text/OCR, solve a photographed task, inspect a screenshot, compare images, extract table data, or otherwise understand image content."
---

# Image Analysis Skill

Use the `analyzeImage` tool when the user asks about attached images or sends images without a text prompt.

## Workflow

1. Read the current user message and collect the `media_id` values from attached image markers.
2. Infer the user's image-related request from the prompt after the markers.
3. Call `analyzeImage(prompt, media_ids)` with the relevant media IDs.
4. Answer from the tool result. If the tool says an image is unavailable, ask the user to reattach it.

## Tool

Use:
- `analyzeImage(prompt, media_ids)` — analyzes one or more images attached by the current user.

Pass only `media_id` values that appear in the conversation context. Never invent media IDs.

## Default Prompt Policy

If the user attached images without a prompt, the inbound preprocessor provides a default prompt asking to describe the images in detail. Use that prompt when calling the tool.

## Scope

Use this tool for image understanding:
- describing images;
- OCR / reading text from images;
- solving photographed equations or tasks;
- extracting data from screenshots, receipts, tables, charts, or documents shown as images;
- comparing several attached images.

Do not use this tool for image generation or image editing. If the user asks to create or modify an image, explain that image generation/editing is not available in this service yet.

