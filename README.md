# Решение среды Pong из набора игр Atari с помощью DQN
![Обученная модель](/assets/video.gif)

## Для улучшения сходимости используются следующие приемы
* Эпсилон-жадный метод - для изучения среды
* Буфер примеров и Целевая сеть - решают проблему iid (данные должны быть независимы и одинаково распределены)

## Алгоритм 
Источники: DeepMind 2013 Playing Atari with Deep Reinforcement Learning, Nature 2015 Human-level control through deep reinforcement learning
1. Инициализировать сеть *Q(s,a)* и целевую сеть *Qt(s, a)*
2. Выбрать действие *а* с помощью эпсилон-жадного метода
3. Выполнить действие *а* и получить награду *r* и следующее состояние *s'*
4. Сохранить переход *(s, a, r, s')* в буфер примеров
5. Выбрать случайный обучающий набор из буфера примеров
6. Для каждого перехода в выбранном наборе вычислить целевое значение *y = r*, если эпизод закончился на данном шаге, иначе *y = r + GAMMA*max(Qt)*
7. Рассчитать потери *L = (Q - y)^2*
8. Обновить *Q*
9. Каждые *N* шагов копировать параметры из *Q* в *Qt*
10. Повторять с шага 2, пока сходимость не будет достигнута

## Обертки
Одни повышают скорость обучения (масштабирование кадров, стек кадров), другие исправляют особенности платформы (эффект мерцания)
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py

## Установка
pip install -r requirements.txt

## Запуск
Обучение запускает train.py

```bash
$ python train.py --help
usage: train.py [-h] [--cuda] [--env ENV] [--reward REWARD]

optional arguments:
  -h, --help       show this help message and exit
  --cuda           Enable cuda
  --env ENV        Name of the envitoment, default=PongNoFrameskip-v4
  --reward REWARD  Mean reward boundary for stop of training, default=19.50
```

Запустить 1 эпизод с обученной моделью

```bash
$ python play.py --help
usage: play.py [-h] -m MODEL [-e ENV] [-r RECORD] [--no-visualize]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model file to load
  -e ENV, --env ENV     Enviroment name to use, default=PongNoFrameskip-v4
  -r RECORD, --record RECORD
                        Directory to store video recording
  --no-visualize        Disable visualization of the game play
  ```

### Пример 1. Запуск тренировки
```bash
$ python train.py --cuda
```

### Пример 2. Запуск среды с тренированной моделью
```bash
$ python play.py --model PongNoFrameskip-v4-best.dat
```

## Ошибки
Exception: ROM is missing for pong
ValueError: too many values to unpack
```bash
$ pip install gym[atari,accept-rom-license]==0.21.0
```

AttributeError: module 'gym.envs.atari' has no attribute 'AtariEnv'
```bash
$ pip uninstall ale-py
$ pip install ale-py==0.7.5
```

## Результаты
Одна видеокарта GeForce GTX 1650.
На обучение ушло 3,5 часа.
Среднее вознаграждение 18 (85% побед) достигнуто за 660 тыс. шагов.

![Средняя награда за 100 шагов](/assets/result.png)