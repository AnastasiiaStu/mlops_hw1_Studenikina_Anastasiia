# mlops\_hw1\_Studenikina\_Anastasiia

## Структура проекта
```

mlops\_hw1\_Studenikina\_Anastasiia/

├── data/
│   ├── raw/
│   │   ├── .gitignore
│   │   └── wine.csv.dvc
│   └── processed/
│       └── .gitignore
├── src/
│   ├── prepare.py
│   └── train.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md

```



## Запуск - или короткое описание действий
### 1. Клонирование репозитория

```bash
git clone https://github.com/AnastasiiaStu/mlops\_hw1\_Studenikina\_Anastasiia.git
cd mlops\_hw1\_Studenikina\_Anastasiia
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt   # ко всем командам pip и dvc добавляла python -m (иначе они не запускались)
```

### 3. Получение данных

```bash
dvc pull
```

### 4. Запуск пайплайна

```bash
dvc repro
```

### 5. Просмотр результатов в MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Открываем браузер и вводим http://localhost:5000

## Проверка воспроизводимости

```bash
git clone https://github.com/AnastasiiaStu/mlops\_hw1\_Studenikina\_Anastasiia.git
cd mlops\_hw1\_Studenikina\_Anastasiia
pip install -r requirements.txt
dvc pull
dvc repro
```
Запускала 2 или 3 раза - работает стабильно

## Загруженный Датасет

*\*Wine Recognition Dataset\*\* из scikit-learn:

- 178 образцов
- 13 признаков (химический состав)
- 3 класса (сорта вин)
- Разделение: 142 train / 36 test

## Ссылка на итоговый репозиторий - на всякий случай еще раз
https://github.com/AnastasiiaStu/mlops\_hw1\_Studenikina\_Anastasiia



