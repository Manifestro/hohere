# hohere

[English](README.md) | **Русский**

Набор инструментов для генерации, синтеза, проверки и публикации **синхронизированных
двухканальных датасетов разговорной речи** для полнодуплексных моделей устного диалога.

```
Scenario plan ─► Qwen dialogue generator ─► dialogue validator ─► Kokoro TTS
             ─► timeline composer ─► audio QC ─► manifest ─► local export / private HF repo
```

- Диалоги пишет **Qwen** (любой OpenAI-совместимый сервер: vLLM, SGLang, DashScope, Ollama).
- Речь синтезирует **Kokoro-82M**, 24 кГц моно, у каждого собеседника постоянный голос.
- Композитор расставляет паузы, наложения, бэкченнелы, перебивания, обрывы и самоисправления
  с **целочисленными временными метками с точностью до сэмпла**, детерминированно по
  записанному seed.
- Результат: `channel_a.wav` (только собеседник A) + `channel_b.wav` (только собеседник B)
  одинаковой длины, плюс `events.json` и `metadata.json` для каждого разговора.
- Каждый этап можно возобновить, отклонённые элементы всегда сохраняются с причинами в
  машиночитаемом виде, разбиение на выборки фиксируется по семействам сценариев до синтеза.

Статус: экспериментальный (Этап 1 — работающий конвейер).

## Состав датасета

Диалоги планируются по их **основному разговорному навыку** (другие виды поведения внутри
них тоже могут встречаться):

| Навык | Доля | Часов по умолчанию (из 20) |
|---|---:|---:|
| `simple_dialogue` — вопросы, ответы, уточнения | 40% | 8 |
| `context_memory` — отсылки на несколько реплик назад | 20% | 4 |
| `turn_taking` — паузы, колебания, ожидание | 15% | 3 |
| `backchannel` — «угу», «ага» во время речи собеседника | 10% | 2 |
| `interruption` — остановить, уточнить, перенаправить | 10% | 2 |
| `incomplete_thoughts` — самоисправление, продолжение мысли | 5% | 1 |

Доли задаются в `dataset.skill_mix` в конфиге.

## Установка

```bash
uv sync                      # ядро (планирование, проверка, композитор, mock-провайдеры)
uv sync --extra kokoro       # + Kokoro TTS (нужен системный пакет `espeak-ng`)
uv sync --extra hf           # + экспорт и публикация на Hugging Face
uv sync --all-extras         # всё (GPU-под)
```

## Быстрый старт (офлайн, без моделей)

```bash
uv run hohere pipeline --config configs/mock.yaml
uv run hohere export   --config configs/mock.yaml --out exports/mock
```

## Настоящий запуск

Запустите сервер Qwen (см. [docs/runpod.md](docs/runpod.md)), затем:

```bash
export HOHERE_QWEN_BASE_URL=http://localhost:8000/v1
uv run hohere generate   --config configs/smoke.yaml
uv run hohere synthesize --config configs/smoke.yaml
uv run hohere compose    --config configs/smoke.yaml
uv run hohere validate   --config configs/smoke.yaml
# или всё сразу:
uv run hohere pipeline   --config configs/smoke.yaml
uv run hohere status     --config configs/smoke.yaml
```

Любой этап можно прервать в любой момент; повторный запуск продолжит с места остановки,
не дублируя разговоры.

## Экспорт и публикация

```bash
uv run hohere export  --config configs/default.yaml --out exports/v0.1.0
uv run hohere export  --config configs/default.yaml --out exports/v0.1.0-train --no-text
HF_TOKEN=... uv run hohere publish --repo-id namespace/dataset-name --export-dir exports/v0.1.0 --tag v0.1.0
```

Публикация по умолчанию **приватная** и отказывается загружать в уже существующий публичный
репозиторий. Токен берётся из `$HF_TOKEN` (или `hf auth login`) и никогда не записывается
на диск.

Экспорт напрямую загружается через 🤗 Datasets:

```python
from datasets import load_dataset
ds = load_dataset("namespace/dataset-name")   # или локальная папка экспорта
row = ds["train"][0]
row["channel_a"], row["channel_b"], row["events"]
```

## CLI

| Команда | Что делает |
|---|---|
| `hohere init [path]` | записывает полный конфиг по умолчанию |
| `hohere plan` | строит и показывает план (спецификации разговоров + неизменяемые выборки) |
| `hohere generate` | диалоги Qwen → проверка → `dialogues/accepted` / `dialogues/rejected` |
| `hohere synthesize` | сегменты Kokoro с кэшем фрагментов и повторными попытками |
| `hohere compose` | двухканальные таймлайны + проверка качества аудио |
| `hohere validate` | структурные и аудиопроверки всего датасета → `reports/validation.json` |
| `hohere manifest` | манифест релиза с SHA-256 каждого файла |
| `hohere export` | HF-совместимый экспорт в parquet с карточкой датасета |
| `hohere publish` | загрузка в приватный датасет-репозиторий HF, проверка, тег |
| `hohere pipeline` | generate → synthesize → compose → validate |
| `hohere status` | прогресс каждого этапа |

## Python API

```python
from hohere.config import load_config
from hohere.pipeline import Pipeline

cfg = load_config("configs/smoke.yaml")
pipe = Pipeline(cfg)
pipe.generate(limit=10)
pipe.synthesize()
pipe.compose()
report = pipe.validate()
```

Каждый этап можно использовать и отдельно — см. [examples/](examples/) и
[docs/dataset_format.md](docs/dataset_format.md).

Покадровая разметка (активность и события смены реплик с частотой 12,5 Гц) и наборы
оценочных тестов `hohere-bench` с протоколами оценки:
[docs/bench-and-labels.md](docs/bench-and-labels.md).

## Разработка

```bash
uv run pytest                    # юнит-тесты + офлайн интеграционные тесты
HOHERE_RUN_EXTERNAL=1 uv run pytest -m external   # настоящие Qwen / Kokoro / HF (по желанию)
uv run ruff check . && uv run ruff format --check .
```

Сгенерированные данные (`data/`, `exports/`), кэши и учётные данные исключены из git.

## Лицензия

MIT
