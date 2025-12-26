# Neural Network Data Processing

Лабораторные работы по курсу **"Нейронные сети обработки данных"**.

## Содержание

| Лабораторная | Тема | Описание |
|--------------|------|----------|
| [Lab1](Lab1/) | Нейронная сеть Хопфилда | Автоассоциативная сеть с обратными связями для распознавания образов |
| [Lab2](Lab2/) | Многослойный персептрон | Сеть с прямым распространением, алгоритм обратного распространения ошибки |
| [Lab3](Lab3/) | Сеть РБФ | Радиально-базисные функции для классификации |
| [Lab4](Lab4/) | Конкурентная нейронная сеть | Самоорганизующаяся сеть для кластеризации |

## Структура проекта

```
Neural_network_data_processing/
├── Doc/
│   ├── Theory.md              # Общие теоретические сведения об ИНС
│   ├── PDF JPG/               # Изображения из методички
│   └── ...
├── Lab1/
│   ├── description.md         # Описание лабораторной работы
│   ├── solution.cpp           # Решение на C++
│   └── report.tex             # Шаблон отчета
├── Lab2/
│   ├── description.md
│   ├── solution.cpp
│   └── report.tex
├── Lab3/
│   ├── description.md
│   ├── solution.cpp
│   └── report.tex
├── Lab4/
│   ├── description.md
│   ├── solution.cpp
│   └── report.tex
└── README.md
```

## Теоретические сведения

Общие теоретические сведения о нейронных сетях находятся в файле [Doc/Theory.md](Doc/Theory.md), включая:
- Введение в ИНС
- Биологический и искусственный нейрон
- Функции активации
- Классификация нейронных сетей

## Типы нейронных сетей

### Сеть Хопфилда (Lab1)
- **Тип:** Автоассоциативная, с обратными связями
- **Обучение:** Без учителя (правило Хебба)
- **Применение:** Ассоциативная память, распознавание зашумленных образов

### Многослойный персептрон (Lab2)
- **Тип:** С прямым распространением
- **Обучение:** С учителем (backpropagation)
- **Применение:** Классификация, аппроксимация функций

### Сеть РБФ (Lab3)
- **Тип:** С прямым распространением, радиально-базисные функции
- **Обучение:** Гибридное (кластеризация + градиентный спуск)
- **Применение:** Классификация при хорошей кластеризации данных

### Конкурентная сеть (Lab4)
- **Тип:** Самоорганизующаяся
- **Обучение:** Без учителя (конкурентное обучение)
- **Применение:** Кластеризация, сжатие данных

## Литература

1. **Aleksander I., Morton H.** An Introduction to Neural Computing. — London: Chapman & Hall, 1990.

2. **Головко В.А.** Нейронные сети: обучение, организация и применение. Учеб. пособие для вузов. — М.: ИПРЖР, 2001. — 256 с.

3. **Bishop C.M.** Neural Networks for Pattern Recognition. — Oxford: Clarendon Press, 1995. — 482 p.

4. **Hopfield J.J.** Neural networks and physical systems with emergent collective computational abilities // Proc. Natl. Acad. Sci. USA. — 1982. — Vol. 79. — P. 2554.

5. **Kohonen T.** Self-organization and associative memory. — Springer-Verlag, 1989. — 312 p.

6. **Kohonen T.** Self-organized formation of topologically correct feature maps // Biol. Cybernetics. — 1982. — Vol. 43. — P. 56-69.

7. **Kohonen T.** Self-organizing maps. — Springer-Verlag, 1995. — 362 p.

8. **Rumelhart D.E., Hinton G.E., Williams R.J.** Learning internal representation by error propagation: McClelland J.L. and Rumelhart D.E. (Eds). Parallel Distributed Processing: Exploration in the Microstructure of Cognition. — MIT Press, Cambridge MA. — 1986. — Vol. 1. — P. 318-362.

9. **Хайкин С.** Нейронные сети: полный курс, 2-е изд.: Пер. с англ. — М.: Издательский дом «Вильямс», 2006. — 1104 с.

10. **Ежов А.А., Шумский С.А.** Нейрокомпьютинг и его применение в экономике и бизнесе. — М.: Мир, 1998. — 222 c.

## Требования

- **Язык программирования:** C/C++
- **Размер образов:** 
  - Lab1: 10×10 (бинарные/биполярные)
  - Lab2-4: 6×6

## Лицензия

MIT License
