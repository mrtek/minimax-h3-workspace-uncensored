---
name: storyboard
description: Превращает рассказанную историю («первый кадр то-то, второй кадр то-то») в промпт для ОДНОЙ картинки-раскадровки: широкий кадр 16:9, разбитый на 2-9 кадров подряд, без номеров, подписей, стрелок и любого текста. Выдаёт готовый промпт под выбранную модель (Krea 2, FLUX.1 dev или Qwen 2.1). Включай, когда чат настроен на «Раскадровка».
---

# Раскадровка

Результат: одна картинка 16:9, разделённая на N кадров. Кадры идут по порядку (слева направо, сверху вниз) и рассказывают историю. На картинке ТОЛЬКО рисунки: никаких номеров, подписей, заголовка, стрелок, плашек, пометок про камеру или объектив, никаких заметок режиссёра. Кадры занимают ВСЮ картинку до самых краёв: никаких полей и внешней рамки вокруг листа. Между кадрами только очень тонкая линия (1-2 мм) или они идут стык в стык (по просьбе пользователя; по умолчанию тонкая тёмная линия).

Про видео, монтажные листы, длительность и перенос в видеогенераторы здесь не пиши ничего: это только картинка.

## Шаг 1. Разбор истории

Пользователь пишет историю свободно, например: «первый кадр: девочка на пороге дома, второй: она идёт по лесу, третий: находит лису». Сделай так:
- Число кадров N: сколько кадров назвал пользователь; если названо число («на 6 кадров»), бери его; если ни того ни другого, выбери 6. Допустимо от 2 до 9. Если просят больше 9, скажи одной строкой, что на одной картинке лучше не больше 9, и сделай первые 9 (или предложи две картинки по 6-9).
- Содержание каждого кадра бери из слов пользователя без изменений по смыслу. Если он описал только часть кадров, недостающие допиши коротко и логично, не меняя сюжет.
- Чередуй крупность (общий, средний, крупный план), если пользователь не задал иначе: после общего плана хорошо идёт крупный. Два одинаковых плана подряд не ставь.

## Шаг 2. Замки (чтобы кадры выглядели как одна история)

1. **Персонаж**: короткое описание (возраст словами этапа жизни, причёска, главная одежда, цвета) и ОДИНАКОВОЕ слово в слово везде, где он появляется. Не перегружай деталями.
2. **Стиль**: один на все кадры («soft watercolor illustration», «realistic cinematic photograph», «flat vector cartoon», «3D animated film still»). Если пользователь стиль не назвал, выбери подходящий по сюжету и скажи об этом одной строкой.
3. **Свет и палитра**: одни на весь лист (время суток, настроение, 2-3 главных цвета).
4. **Место**: если сцена в одном месте, держи одинаковые детали фона.

## Шаг 3. Сетка по числу кадров

| N | Сетка |
|---|---|
| 2 | два кадра рядом |
| 3 | три кадра в один ряд |
| 4 | 2 колонки и 2 ряда |
| 5 | три кадра сверху и два снизу |
| 6 | 3 колонки и 2 ряда |
| 7 | четыре сверху и три снизу |
| 8 | 4 колонки и 2 ряда |
| 9 | 3 колонки и 3 ряда |

Фраза про сетку и разделение (подставляй число словами: "four", "six" и т.д.; без числа модель иногда рисует лишнее):
- с линией: "divided into six equal frames in a grid of three columns and two rows, a full-bleed layout where the pictures extend all the way to the outer edges of the image, so that the top, bottom, left and right sides of the image are picture content with no white margin at all, and the pictures are separated from each other only by very thin black gutters";
- стык в стык: "divided into six equal frames in a grid of three columns and two rows, the pictures touching edge to edge with no gaps and no borders, extending all the way to the outer edges of the image with no white margin at all".

## Шаг 4. Выдача под модель

Модель подсказывает приложение (в начале сообщения написано, какая выбрана). Одной строкой по-русски скажи: сколько кадров получилось и какая сетка. Затем один английский блок prompt. Русскую версию, негатив и настройки не выводи.

Общие правила для всех: текст и цифры на картинку не просим и не упоминаем слов вроде "label", "caption", "number" (модель их нарисует). Отсутствие лишнего описываем утверждением: "The sheet consists only of the illustrated frames, nothing is printed on it."

### Krea 2 (один абзац, до 1500 символов, без негатива)
Носитель первым словом, в начале: "A storyboard sheet, [STYLE], ..."

```prompt
A storyboard sheet in a single wide 16:9 image, [STYLE], divided into six equal frames in a grid of three columns and two rows, a full-bleed layout where the pictures extend all the way to the outer edges of the image, so that the top, bottom, left and right sides of the image are picture content with no white margin at all, and the pictures are separated from each other only by very thin black gutters, read left to right and top to bottom. Frame one: [wide shot of the scene]. Frame two: [medium shot...]. Frame three: [close-up...]. Frame four: [...]. Frame five: [...]. Frame six: [...]. The same [CHARACTER SHORT] appears in every frame with identical face, hair and outfit. Consistent [LIGHT AND PALETTE] across all frames. The sheet consists only of the illustrated frames, nothing is printed on it.
```

### FLUX.1 [dev] (80-220 слов, порядок слов важен, без негатива)
Самое важное первым: носитель, число кадров и сетка, потом кадры по порядку, потом общие замки.

```prompt
A storyboard sheet, [STYLE], one wide image divided into six equal frames in a grid of three columns and two rows, a full-bleed layout where the pictures extend all the way to the outer edges of the image, so that the top, bottom, left and right sides of the image are picture content with no white margin at all, and the pictures are separated from each other only by very thin black gutters. Frame one, top left: [description]. Frame two, top middle: [...]. Frame three, top right: [...]. Frame four, bottom left: [...]. Frame five, bottom middle: [...]. Frame six, bottom right: [...]. The same [CHARACTER SHORT] in every frame, identical face, hair and clothing. [LIGHT AND PALETTE] across the whole sheet. Only illustrated frames, nothing is printed on the sheet.
```

### Qwen Image 2.1 (наблюдательный стиль, 150-300 слов, настоящее время)
Негатив в приложении вписывается отдельным полем, блок negative НЕ выводи. Слова "horizontal/landscape" про кадр не пиши, размер задаётся в приложении.

```prompt
The image is a [STYLE] storyboard sheet divided into six equal frames in a grid of three columns and two rows, a full-bleed layout where the pictures extend all the way to the outer edges of the image, so that the top, bottom, left and right sides of the image are picture content with no white margin at all, and the pictures are separated from each other only by very thin black gutters. In the top left frame, [description]. In the top middle frame, [...]. In the top right frame, [...]. In the bottom left frame, [...]. In the bottom middle frame, [...]. In the bottom right frame, [...]. The same [CHARACTER SHORT] appears in every frame with the same face, hair and clothing. The lighting and colors are consistent across all frames: [LIGHT AND PALETTE]. The sheet contains only the six illustrated frames and nothing is printed on it.
```

## Режим по кадрам (рекомендуется)

В приложении есть режим «По кадрам» (FLUX.1 dev): каждый кадр рисуется ОТДЕЛЬНОЙ картинкой (сохраняется отдельно в полном размере), а лист с тонкой линией между кадрами склеивает программа. Поэтому здесь никаких рамок, полей и «листа» в промптах нет.

Выдача: один блок prompt, в котором каждый кадр на своей строке: `1: ...`, `2: ...`. Число кадров строго 2, 3, 4, 6, 8 или 9. Каждая строка полный самостоятельный английский промпт ОДНОГО кадра:
- начало строки: "In the same art style as image 1, a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette." (если картинок нет, вместо "image 1" назови стиль словами);
- герой: ссылкой ("one single small girl from image 1") И коротко словами (причёска, одежда, цвета), одинаково во всех строках;
- место: "the location from image 2" или словами; пиши "a new composition, not a copy of image 2";
- план и действие одного кадра, свет.

```prompt
1: In the same art style as image 1, a full-bleed picture that fills the entire frame right to its edges with no white border, no paper margin and no vignette. A wide shot of one single small girl from image 1 (short brown bob, red scarf, green coat) standing on the porch in front of the door of the wooden house from image 2 at dawn; a new composition, not a copy of image 2.
2: In the same art style as image 1, a full-bleed picture ... A close-up of the face of one single small girl from image 1 (short brown bob, red scarf, green coat) looking up in surprise, soft dawn light.
3: ...
4: ...
```

## Если приложены картинки (герой, локация): раздел про референсы

Работает только с Qwen 2.1 и FLUX.2 klein (Krea 2 картинки не принимает). Порядок задаёт пользователь: `<image1>` первая приложенная картинка (обычно герой или лист персонажа), `<image2>` вторая (обычно локация), `<image3>` третья по желанию. Тогда:
- Героя и место словами НЕ описывай: пиши "the character from <image1>" и "the location from <image2>" (для FLUX.2 пиши "image 1", "image 2", без угловых скобок).
- Стиль бери с референса: "in the same art style as <image1>" (или явно назови другой стиль, если пользователь просит).
- Кадры, где локации нет (крупный план героя), пиши без места или с общим фоном из <image2>.
- Одежду и лицо героя в каждом кадре тоже ссылкой: "wearing the same outfit as in <image1>".

Шаблон Qwen 2.1 (то же, что выше, но герой и место ссылками):
```prompt
The image is a storyboard sheet in the same art style as <image1>, divided into six equal frames in a grid of three columns and two rows, a full-bleed layout where the pictures extend all the way to the outer edges of the image, so that the top, bottom, left and right sides of the image are picture content with no white margin at all, and the pictures are separated from each other only by very thin black gutters. In the top left frame, a wide shot of the character from <image1> standing in the location from <image2>. In the top middle frame, [...]. [...] The character from <image1> appears in every frame with the same face, hair and outfit. The lighting and colors are consistent across all frames: [LIGHT AND PALETTE]. The sheet contains only the six illustrated frames and nothing is printed on it.
```

Шаблон FLUX.2 klein (ссылки "image 1", "image 2"). Проверено: FLUX склонен просто копировать картинку локации и «забывать» героя, поэтому героя нужно и ссылкой, и КОРОТКО словами в каждом кадре, а ещё прямо сказать, что каждый кадр новая композиция:
```prompt
A storyboard sheet in the same art style as image 1: four frames in a grid of two columns and two rows. The four pictures run all the way to the very edges of the image with no outer margin and no white border, separated only by hairline-thin black lines. The character from image 1 appears in every one of the four frames and each frame is a new composition, not a copy of image 2. The house and landscape look like image 2. Frame one, top left: a wide shot of [CHARACTER SHORT], standing right in front of the door of [LOCATION SHORT]. Frame two, top right: a close-up of the same character's face [emotion]. Frame three, bottom left: [...] of the same character [action]. Frame four, bottom right: [...] of the same character [action]. Same face, hair and outfit as image 1 in every frame. [LIGHT AND PALETTE]. Only illustrated frames, nothing is printed on the sheet.
```

## Шаг 5. Одной строкой в конце

Предупреди: нейросеть рисует все кадры за раз, поэтому персонаж в разных кадрах может слегка отличаться. Чем проще и короче описание героя, тем лучше он держится. Если кадры получились плохо, можно поменять модель в выборе или упростить описание.

## Ошибки

- Нумерация, подписи, заголовок, плашки, стрелки, термины камеры (lens, mm, pan, tilt) в промпте: убрать.
- Слова про видео, секунды, анимацию: убрать.
- Длинное описание героя в каждом кадре: коротко и одинаково.
- Больше 9 кадров на одном листе.
- Русский текст внутри промпта: промпт всегда на английском.
- Реальные люди, бренды, персонажи франшиз: только обобщённый типаж.
