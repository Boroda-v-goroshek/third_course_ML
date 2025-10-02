***Структура репозитория***:

```
root: вы находитесь тут
    - code: решения задач
    - configs: конфигурации для каждой задачи
    - data: тестовые и трейновые данные для каждой задачи
    - scripts: .sh скрипты для упрощенного пользования этим репозиторием
    - scr: ядро репозитория, переиспользуемый код
```

***Использование***

из вашей директории:

```
wget -q -O - https://astral.sh/uv/install.sh | sh
или
sudo snap install astral-uv
uv venv или uv venv venv
source .venv/bin/activate
```

из root:

```
uv pip install -e .
bash scripts/<task>.sh - общий вид
bash scripts/GenAI-1-06.sh - задача первого блока
bash scripts/DA-2-18.sh - задача второго блока
```
