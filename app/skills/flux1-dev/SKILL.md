---
name: flux1-dev
description: Пишет промпты для флагманской модели FLUX.1 [dev] (12B FP8, Dual-CLIP T5-XXL + CLIP-L, 20-28 шагов, фотореализм высшего класса, поддержка LoRA). Естественный английский язык, детальное описание света, оптики и текстуры кожи. Включай, когда выбран режим «FLUX.1 [dev]».
---

# FLUX.1 [dev] 12B: промпт для генерации и правки изображений

Модель FLUX.1 [dev] использует архитектуру с 12 миллиардами параметров и двойной текстовый энкодер (T5-XXL + CLIP-L), что обеспечивает высочайшее качество понимания естественного языка, непревзойдённый фотореализм, чёткую анатомию пальцев и реалистичные текстуры кожи.

---

## 1. Порядок слов и стиль описания

Модель лучше всего понимает связную художественную прозу на английском языке (без тегов и весов скобок):
1. **Жанр и носитель**: `A hyper-realistic 35mm photograph`, `Cinematic film still shot on ARRI Alexa`, `Studio editorial fashion photo`.
2. **Главный объект**: типаж, точный возраст, телосложение, поза, выражение лица и взгляд.
3. **Анатомия и кожа**: `natural skin texture with visible pores, subtle skin sheen, intricate realistic details`.
4. **Одежда или обнажённость**: детальное описание ткани, посадки, либо `completely unclothed, nude, exposed skin`.
5. **Освещение и камера**: `dramatic chiaroscuro lighting, soft diffused rim light, 85mm f/1.4 lens, shallow depth of field, delicate film grain`.
6. **Окружение и глубина**: передний план, задний план, атмосфера.

---

## 2. Правка изображения (Image-to-Image / Reference Mode)

В режиме правки изображения с FLUX.1 [dev]:
- Первое изображение выступает референсом композиции и лица (`image 1`).
- Дополнительные референсы: `image 2`, `image 3`.
- Чётко разграничивайте, что **сохранить** (черты лица, форму тела, ракурс), и что **изменить** (окружение, одежду, свет).

Пример формулы:
```prompt
Modify image 1 by transforming the background into a sun-drenched Mediterranean villa terrace. Retain the subject's exact facial structure, hair color, and expression from image 1. Natural sunlight casting soft shadows, photorealistic 8k render.
```

---

## 3. Чего избегать
- Не используйте искусственные теги качества вроде `masterpiece, 8k, trending on artstation` (FLUX и так даёт максимум качества).
- Не пишите отрицания вроде `no clothes`, `without shoes`: формулируйте утвердительно (`unclothed, bare feet`).
- Не вставляйте технические параметры разрешения в текст промпта (они задаются в интерфейсе).

---

## 4. Формат выдачи Агента
- Одна строка по-русски с рекомендациями (разрешение, совет по свету).
- Блок ````prompt ... ```` с английским описанием.
