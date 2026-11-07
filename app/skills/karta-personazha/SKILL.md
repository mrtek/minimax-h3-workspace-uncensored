---
name: karta-personazha
description: Превращает вольное описание персонажа в промпт для чистого reference-листа персонажа из 4 панелей (в полный рост спереди и сзади, портрет анфас и в профиль) для Krea 2 и Qwen Image 2.1. Лист фиксирует лицо, одежду и стиль, чтобы персонаж не менялся в следующих картинках и видео. Включай, когда сообщение начинается со слова "персонаж" или просят карту, лист или референс персонажа.
---

# Карта персонажа

Лист из 4 панелей на одной картинке: 1) в полный рост спереди, 2) в полный рост сзади, 3) портрет анфас, 4) портрет в профиль. Один и тот же человек (или существо), одна одежда, один фон, один свет, один масштаб в парах панелей. Ничего лишнего: без палитры, эмоций и сцен, если пользователь не попросил "полный лист".

Готовый лист потом используют как референс: в Qwen (режим "по картинке") и в разделе Референсы для видео. Поэтому лист должен быть чистым и одинаковым по стилю во всех панелях.

## Шаг 1. Четыре замка (всё на английском, плотно и конкретно)

1. **DNA** - неизменяемая геометрия: возраст, форма лица, скулы, челюсть, нос, брови, глаза (цвет, форма), волосы (цвет, длина, стрижка, чёлка, борода), кожа (тон, веснушки, поры), приметы (шрамы, родинки, татуировки), телосложение, рост, осанка. Не клади сюда одежду, эмоцию, позу, свет.
2. **STYLE** - один конкретный стиль: "realistic studio photograph", "soft hand-drawn anime illustration", "flat vector cartoon", "3D animated film render". Если пользователь стиль не назвал, выбери подходящий по описанию и скажи об этом одной строкой.
3. **WARDROBE** - одна фирменная одежда, одинаковая во всех панелях, с материалом и цветом каждой вещи ("black wool blazer", "cream linen shirt", "dusty-blue cotton skirt").
4. **BACKGROUND** - нейтральный студийный фон одного цвета (по умолчанию светло-серый), ровный мягкий свет.

Цвета волос, глаз, кожи и каждой вещи фиксируй для себя и повторяй одинаково. Выдумывай только то, что нужно для листа и не противоречит словам пользователя.

## Шаг 2. Композиция листа

Четыре панели в один ряд, горизонтальный кадр. Панели 1 и 2 - одинаковый масштаб, вся фигура от головы до ног. Панели 3 и 4 - голова и плечи, одинаковый масштаб и свет.

Правило по умолчанию для панели 1 (в полный рост спереди), если персонаж - человек или гуманоид: головы на этой панели НЕТ. Это не манекен и не обрубок шеи: вырез горловины просто пустой, сквозь него виден фон. Так одежда видна целиком и ничто не отвлекает. Панель 2 (сзади) всегда с головой, панели 3 и 4 (портреты) тоже. Исключение: пользователь явно просит голову на панели 1 ("с головой", "покажи лицо и спереди") - тогда панель 1 рисуется с головой. Для животных и существ голову на панели 1 оставляй.

## Шаг 3. Выдача под модель

Если приложение или пользователь назвал модель - выдай один промпт под неё. Если не названа - выдай два блока с подписями Krea 2 и Qwen 2.1.

### Krea 2 (только текст, без негатива)

- Один связный абзац на английском, до 1500 символов. Носитель первым словом.
- Негатива нет, фразы "no X / without X" не работают. Вместо них пиши, что есть: "clean plain light-grey studio backdrop with only the four figures on it".
- Не добавляй "masterpiece, 8k, beautiful".
- Панель 1 без головы Krea не выполняет (проверено двумя формулировками, голова рисуется всё равно). Для Krea панель 1 всегда с головой, пользователю одной строкой скажи, что безголовый вид делает только Qwen по загруженной картинке. Размер холста для Krea 1376x768 (1K).
- Проверенный результат: четыре панели одного стиля, одна одежда, лицо и очки держатся, генерация 20-40 секунд.

Шаблон:
```prompt
Character reference sheet in a single wide image, four panels side by side, [STYLE]. Panel one: full-body front view of [DNA short], standing in a neutral pose facing the viewer, wearing [WARDROBE]. Panel two: the same character in the same pose and outfit seen from behind, same scale as panel one. Panel three: head-and-shoulders portrait facing the viewer, neutral expression, [face details from DNA]. Panel four: head-and-shoulders portrait in profile, same lighting and scale as panel three. The face, hairstyle, skin tone and outfit are identical in all four panels. Even soft studio light, clean plain [BACKGROUND] backdrop with only the four figures on it.
```

### Qwen Image 2.1

- Наблюдательный стиль, 150-300 слов, настоящее время, третье лицо; слово "vertical/horizontal/portrait/landscape" про кадр не пиши, размер задаётся в приложении.
- Возраст словами этапа жизни ("woman in her thirties"), не числом. Цвет кожи и этничность - только если задал пользователь.
- Негатив выдавай отдельным блоком ```negative, не внутри промпта.
- Картинку-референс описывать вручную не нужно, если она загружена: используй "preserve the face, hair and outfit from the reference".

Шаблон:
```prompt
The image is a [STYLE] character reference sheet split into four equal panels side by side on a plain [BACKGROUND] studio backdrop. In the first panel the [character] stands full-body facing the viewer in a neutral pose, [DNA], wearing [WARDROBE]. In the second panel the same character stands in the same pose seen from behind, same scale, the back of the hair and the outfit clearly visible. The third panel is a head-and-shoulders portrait facing the viewer with a neutral expression, showing [face details]. The fourth panel is a head-and-shoulders portrait in profile, with the same lighting and scale as the third. Soft even studio light falls from the front with gentle shadows. The face, hairstyle, skin tone and clothing are identical in every panel, and the composition is balanced and symmetrical.
```
```negative
different face between panels, different hair between panels, different outfit, inconsistent proportions, merged or overlapping figures, scenic background, dramatic lighting, cropped figure, text, labels, captions, color swatches, watermark, blurry, deformed hands, extra limbs
```

### Qwen по загруженной картинке (персонаж уже есть) - проверенный шаблон

Когда у пользователя уже есть картинка персонажа и она загружена в приложение, не описывай внешность словами: ссылайся на `<image1>`. Этот промпт проверен на реальном персонаже, результат: четыре одинаковые по стилю панели, лицо и аксессуары совпали с исходником, панель 1 без головы. Размер холста 1920x1088.
```prompt
A character reference sheet of the character from <image1>, split into four equal panels side by side in one row, all at the same scale, in the same art style as <image1>.
Panel one: full-body front view, the figure has no head at all, not a mannequin and not a neck stump, the collar is simply empty and the plain background shows through the neckline, standing in a relaxed neutral pose, showing the whole outfit from the neckline to the shoes.
Panel two: full-body back view of the same character in the same pose and outfit, seen from behind, head fully visible.
Panel three: close-up head-and-shoulders portrait facing the viewer with a neutral expression.
Panel four: close-up head-and-shoulders portrait in profile, with the same lighting and scale as panel three.
Keep the same face, hairstyle, [accessories], clothing, colors, proportions and art style as in <image1> in every panel.
Plain solid light grey background, soft even lighting, no text, no labels.
```
Подставляй в `[accessories]` только то, что видно на картинке (очки, серьги, шрам). Если нужна голова на первой панели, замени описание панели 1 на: "full-body front view facing the viewer, standing in a relaxed neutral pose". Всегда пиши число панелей словом: без числа модель иногда рисует лишнюю фигуру. Фон меняется одной фразой (white, light grey, black).

## Вариант из 6 панелей (когда пользователь выбрал "6 панелей", "лист на 6", "×6" или приложение это указало)

Это отдельный лист, для видео, где модель путает ракурсы и дорисовывает невидимое (в частности сбоку и в три четверти). Шесть панелей в один ряд, горизонтальный широкий кадр, все высокие, один масштаб у полноростовых:
1) в полный рост спереди, 2) в полный рост в три четверти (повёрнут примерно на 45 градусов), 3) в полный рост сбоку, в профиль, 4) в полный рост сзади, 5) портрет анфас, 6) портрет в три четверти. Одежда, лицо, стиль, свет и фон одинаковые во всех шести. Правило про голову на панели 1 то же, что выше (по умолчанию без головы только у Qwen по картинке; для Krea голова есть всегда).

Если пользователь просит зеркальные виды ("слева и справа"), напиши панель 3 как "side view facing left", а панель 6 как "three-quarter view facing right", так оба направления присутствуют. Не добавляй больше шести панелей: число панелей всегда пиши словом "six".

### Krea 2, 6 панелей (формат 21:9, один абзац на английском до 1500 символов, без негатива)

```prompt
Character reference sheet in a single very wide image, six tall panels side by side, [STYLE]. Panel one: full-body front view of [DNA short], standing in a neutral pose facing the viewer, wearing [WARDROBE]. Panel two: the same character full-body in a three-quarter view, turned about forty-five degrees, same pose and outfit. Panel three: the same character full-body in a side view in profile, same pose and outfit. Panel four: the same character full-body seen from behind, same pose and outfit. Panel five: head-and-shoulders portrait facing the viewer, neutral expression, [face details from DNA]. Panel six: head-and-shoulders portrait in three-quarter view, same lighting and scale as panel five. All four full-body figures have the same scale, and the face, hairstyle, skin tone and outfit are identical in all six panels. Even soft studio light, clean plain [BACKGROUND] backdrop with only the six figures on it.
```
Предупреди одной строкой: Krea работает примерно в 1 МП, на широком листе панели получаются небольшими; для чёткого листа лучше раздел "Правка картинки" → "Лист персонажа" → 6 панелей (по готовой картинке персонажа).

### Qwen по загруженной картинке, 6 панелей (размер холста 2112x960)

```prompt
A character reference sheet of the character from <image1>, split into six equal tall panels side by side in one row, all at the same scale, in the same art style as <image1>.
Panel one: full-body front view facing the viewer, standing in a relaxed neutral pose, showing the whole outfit from head to shoes.
Panel two: full-body three-quarter view of the same character, turned about forty-five degrees to the left, same pose and outfit, head fully visible.
Panel three: full-body side view in profile facing left, same pose and outfit, head fully visible.
Panel four: full-body back view of the same character, same pose and outfit, seen from behind, head fully visible.
Panel five: close-up head-and-shoulders portrait facing the viewer with a neutral expression.
Panel six: close-up head-and-shoulders portrait in three-quarter view, with the same lighting and scale as panel five.
Keep the same face, hairstyle, [accessories], clothing, colors, proportions and art style as in <image1> in every panel.
Plain solid light grey background, soft even lighting, no text, no labels.
```

## Шаг 4. Что дальше (одной строкой в конце)

Скажи пользователю: сохранить лист в Библиотеку, потом загрузить его как референс в Qwen ("по картинке") или в раздел Референсы для видео. Для самой сцены с этим персонажем пиши новый промпт с фразой "preserve the face, hair and outfit from the reference".

## Полный лист (только по явной просьбе: "полный лист", "с палитрой", "с эмоциями", "со сценами")

Добавь к четырём панелям: ряд подписанных образцов цвета (волосы, глаза, кожа, каждая вещь), 4-6 портретов с эмоциями и 2-3 маленькие сцены. Подписи короткие, по-английски, в кавычках. Остальные правила те же.

## Ошибки

- Палитра, эмоции и сцены без просьбы - не добавлять.
- Разный свет, масштаб или фон между панелями - писать одинаково для всех.
- Одежда или эмоция в DNA - выносить в WARDROBE или убирать.
- Негатив внутри промпта Krea или Qwen - Krea негатива не имеет, у Qwen он отдельным блоком.
- Русский текст внутри промпта - промпт всегда на английском.
- Реальные люди, бренды, персонажи франшиз - только обобщённый типаж.
