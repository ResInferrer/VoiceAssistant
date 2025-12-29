# Game-Agnostic Cognitive Player (GACP)
LLM-агент, который ведет себя как игрок, а не как NPC.

## [О проекте + roadmap](https://docs.google.com/document/d/1EvqQ1nDP_K-V4drSLzz3AxfX93gLw4JFqVFeqKXyCfk/edit?tab=t.0)
**Кратко:**
ЦЕЛЬ ПРОЕКТА: Разработать архитектуру LLM-агента, способную к переносу навыков (transfer learning) между различными средами,  максимально и практично минимизируя ручную адаптацию под каждую новую игру. 
 

Чтобы увидеть архитектуру в графическом виде скопируйте весь код из файла `agent_workflow_architecture.mmd` и вставьте на сайте [Mermaid Live Editor](https://mermaid.live/).
## Установка

> [!IMPORTANT]
> **Обязательные требования:**
> **OS** = **Windows;**
> **RAM** = **16+ GB.**

1. Клонируйте реопзиторий
```bash
git clone https://github.com/KotingGG/Lerche.git
cd Lerche
```

2. Зависимости:
- [Python](https://www.python.org/downloads/)
- Для запуска локальной LLM [Ollama](https://ollama.com/download)
- для удобства и быстроты устанавливки зависимостей `uv`:
```bash
pip install uv
```
- Для установки всех зависимостей из lock-файла:
```bash
uv sync
```

3. Установка локальной модели. [Почему модель `qwen2.5:7b-instruct-q4_0?`](#почему-модель-qwen257b-instruct-q4_0):
```bash
ollama pull qwen2.5:7b-instruct-q4_0
```


## Использование

```bash
uv run main.py
```

## Структура проекта

- `main.py`: Скрипт для запуска модели.
- `graph/`: Логика langgraph агентов. Распознание речи, синтез речи и т.д.
- `confing/`: Конфигурация моделей. Параметры (температура и т.д.), название модели, системный промт.
- `vosk_models/`: Локальная модель для распознания речи. Установлена с [официального сайта vosk](https://alphacephei.com/vosk/models).

## Лицензия
Этот проект распространяется под лицензией MIT.