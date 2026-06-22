---
name: image-generation
description: "Create or edit images with the built-in image generation tool. TRIGGER when: the user asks to create, generate, draw, render, make an illustration/image, edit/change/process an attached or generated image, or follows up on a recent image with requests such as changing colors, labels, text, style, adding/removing objects, fixing a failed edit, or trying again."
---

# Image Generation Skill

Use `generateImage` for image creation and image editing.

## Workflow

1. For a new image, call `generateImage` without `source_media_ids`.
2. For edits or follow-ups, find the relevant image marker in current or recent context: `[Attached image: media_id="..."]` or `[Generated image: media_id="..."]`.
3. Pass the relevant `media_id` values as `source_media_ids`. Prefer the latest relevant generated image unless the user clearly refers to another image.
4. For edits, describe the requested change and what must stay unchanged.
5. Ask one short clarification only if the target image is unclear.

## Result Policy

- Do not claim an image was created or changed unless `generateImage` returned `success: true`.
- Do not invent or manually write markdown image links, `media_id` values, or `[Generated image: ...]`.
- If `generateImage` returns an error or quota limit, say that the image was not created or changed and briefly mention the reason.
