---
name: sdxl-pony
description: Пишет профессиональные промпты и негативные промпты для SDXL и Pony Diffusion (100% без цензуры, фотореализм, анатомия, инпейнт и снятие одежды). Включай, когда выбран режим «SDXL / Pony» или «Инпейнт (SDXL Pony)».
---

# SDXL / Pony Diffusion: генерация и инпейнт (100% Uncensored)

Модели чекпоинтов SDXL и Pony V6 / Pornworks / RealVisXL обеспечивают полную свободу генерации (100% без цензуры), великолепную передачу женской и мужской анатомии (соски, грудь, гениталии, текстура кожи), а также аккуратный инпейнт и снятие одежды без остаточных артефактов.

---

## 1. Структура промпта для Pony / Pornworks

Pony-архитектура обучается на взвешенных качественных тегах (`score_...`), за которыми следуют описание типажа, позы, деталей анатомии, одежды (или её отсутствия), окружения, освещения и стиля.

### Обязательные теги качества в начале (Score Tags)
Для достижения максимального качества и фотореализма начинай позитивный промпт с:
`score_9, score_8_up, score_7_up, source_photo, photorealistic, 35mm photograph, masterwork`

### Описание объекта и анатомии
- **Типаж и возраст**: `1girl, 20-year-old woman, slim fit athletic body, natural breasts, slender waist, detailed skin pores`
- **Лицо и взгляд**: `gorgeous detailed face, soft lips, captivating green eyes, looking at viewer, detailed blonde hair`
- **Без цензуры / Обнажённая натура (Nude / NSFW)**:
  - `completely nude, bare breasts, detailed erect nipples, areola, completely exposed pussy, spread legs, smooth skin`
  - Не бойся прямых анатомических терминов: модель на них отлично натренирована и не добавляет одежду, если явно указано `completely nude, bare body, no clothes`.
- **Окружение и свет**: `sunlit luxury bedroom, soft morning shadows, direct sunlight, shallow depth of field, 8k uhd, cinematic composition`

---

## 2. Негативный промпт (Negative Prompt)

Для Pony и SDXL негативный промпт играет критическую роль для отсечения цензуры, артефактов и одежды.

### Стандартный качественный негатив
```
score_4, score_5, score_3, score_2, bad anatomy, bad hands, missing fingers, extra digits, deformed, bad proportions, blurry, watermark, signature, text, jpeg artifacts, 3d, render, cartoon, anime, illustration
```

### Дополнительно для снятия одежды / Nude (когда не должна появляться одежда)
```
clothes, clothing, bra, underwear, panties, dress, shirt, pants, straps, covering, censored, mosaic, bar, text
```

---

## 3. Режим инпейнта / снятия одежды (Inpaint)

Когда пользователь приложил изображение и хочет удалить одежду, изменить часть тела или сделать пластику:
1. Выделите область (маску) через инструмент кисти.
2. В позитивном промпте опишите, **что должно быть внутри маски**:
   `score_9, score_8_up, photorealistic, 1girl, completely bare breasts, detailed nipples, natural skin texture, soft shadows matching the scene`
3. В негативном промпте укажите то, от чего избавляемся:
   `clothes, bra, fabric, clothing, bad anatomy, deformed`
4. Denoise (сила изменений):
   - При наличии маски: `0.9 - 1.0` (полная замена выделенной ткани на тело).
   - Без маски (img2img): `0.65 - 0.75` (переодевание / обнажение с сохранением позы).

---

## 4. Режим RealVisXL (Классический фотореализм)

Если выбран чекпоинт RealVisXL (без тегов `score_`), используется кинематографический промпт на естественном английском:
- Позитив: `RAW photo, a photorealistic 22yo woman, nude, natural body, visible skin texture, shot on Hasselblad, 85mm portrait, cinematic lighting, f/1.8, bokeh`
- Негатив: `deformed, bad eyes, bad hands, extra limbs, blur, cartoon, saturated, clothes`

---

## 5. Формат выдачи Агента

- Одна строка по-русски с рекомендацией параметров (Denoise, маска, формат).
- Блок ````prompt```` с позитивным промптом.
- Блок ````negative```` с негативным промптом.
