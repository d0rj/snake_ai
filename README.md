# AI for Snake

## Описание

Репозиторий для создания ИИ для змейки с помощью нейронных сетей. Основным фреймворком выбран Keras.

Игра поддерживает удобное и быстрое добавление и использование новых алгоритмов игры. См. раздел **Запуск**.

## Запуск

Запуск алгоритма передвижения по кругу с оконным отображением.

```bash
git clone https://github.com/d0rj/snake_ai.git
```

```bash
cd ./snake_ai/
python main.py
```

### Запуск со своим алгоритмом

Функция алгоритма вызывается каждый ход перед передвижением змейки.

Она должна иметь следующую сигнатуру: 

- *На входе*:
    - объект `engine.Game`, содержащий информацию о текущем состоянии игры;
- *На выходе*:
    - объект перечисления `Direction`, обозначающий новое абсолютное направление движения змейки;
- *Имя*:
    - Может быть любым, по умолчанию ищется имя `snake_ai`;
- *Модуль* (где функцию писать):
    - Можно писать где угодно, но по умолчанию она ищется в модуле `ai.snake_ai`.

Примеры можно посмотреть в папке [ai](https://github.com/d0rj/snake_ai/tree/main/ai) репозитория.

Для указания пути до функции ИИ используй параметр **-m**/**--module**.

Для указания имени функции ИИ (если оно вдруг другое) используй параметр **-f**/**--function**.

```bash
python main.py -m path.to.module -f 'my_awesome_ai_function' # пример запуска программы со своим алгоритмом
```

### Отображение

По умолчанию процесс игры отображается в открытом окне. Для того, чтобы игра происходила в 'тихом режиме' (без окна), следует при запуске программы указать флаг **-n**/**--no-window**:

```bash
python main.py -m path.to.module --no-window
```

### Игра клавиатурой

Можно играть самому с использованием клавиатуры, если при запуске указать флаг **-k**/**--keyboard**:

```bash
python main.py -k
```

## Модели

### [1](https://github.com/d0rj/snake_ai/blob/main/nn1.py)

Последовательная нейронная сеть, обучаемая лишь на выживание.

В итоге выполняет свою задачу отлично: просто вертится вокруг самой себя и живёт вечно.

### [2](https://github.com/d0rj/snake_ai/blob/main/nn2.py)

Последовательная НС, обучаемая на выживание + съедание еды.
