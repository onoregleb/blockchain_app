## Ethereum Wallet Clustering Dashboard — это веб-приложение на основе Streamlit для первичного анализа и кластеризации кошельков в сети Ethereum за последние 3 месяца на основе ончейн-данных. 

## Приложение позволяет:

* Собрать данные транзакций указанного ERC-20 токена через API Etherscan или загрузить подготовленный csv-датасет.
* Выполнить исследовательский анализ данных (EDA): посмотреть основные статистики и распределения.
* Определить оптимальные параметры для кластеризации (метод локтя, Silhouette Score, Davies-Bouldin Index).
* Провести кластеризацию кошельков (KMeans) и визуализировать результаты (PCA-иллюстрация, распределение по кластерам).
* Сгенерировать описание полученных кластеров в реальном времени с помощью GigaChat API (ChatGPT-подобный сервис).

### Основные возможности

1. Сбор данных: получение ежедневных транзакций за выбранный период (до 365 дней). Автоматическая обработка лимитов Etherscan (10k транзакций в день).

2. Исследовательский анализ (EDA): табличное и графическое представление распределений метрик кошельков.

3. Автоматический подбор k: анализ кривой инерции, силуэт-метрики и индекса Davies-Bouldin.

4. Кластеризация и визуализация: запуск KMeans, просмотр статистики по кластерам и визуализация через PCA и графики Matplotlib/Plotly.

5. AI-описание кластеров: отправка статистики по кластерам в GigaChat для генерации человекочитаемого описания.

## Технологический стек

* Python (3.10+)
* Streamlit — веб-интерфейс
* Pandas, NumPy — работа с данными
* Scikit-learn — кластеризация и оценка метрик
* Matplotlib, Plotly — визуализация
* Etherscan API — сбор ончейн-данных
* GigaChat API — генерация описаний кластеров

## Установка и запуск

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/onoregleb/blockchainapp.git
   cd blockchainapp
   ```

2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Добавить секреты для API:
 * Создать файл `.streamlit/secrets.toml` со следующими переменными:
     ```toml
     ETHERSCAN_API_KEY = "вашключetherscan"
     GIGACHAT_AUTH_BASIC_VALUE = "Base64(ClientID:ClientSecret)"
     ```

4. Запустить приложение:
   ```bash
   streamlit run streamlit_app.py
   ```

5. Открыть браузер по адресу `http://localhost:8501`.

## Структура проекта

```text
blockchainapp/
├── streamlit_app.py        # Основной Streamlit-скрипт
├── requirements.txt        # Зависимости проекта
├── src/                    # Сбор и обработка ончейн-данных
│   ├── fetch_wallet.py     # Получение транзакций и расчёт метрик для кошельков
│   └── data_example.csv    # Пример набора данных
└── utils/                  # Утилиты для анализа, кластеризации и визуализации
    ├── preprocessing.py    # Предобработка и масштабирование признаков
    ├── eda.py              # Генерация EDA-графиков (распределения, log-преобразование)
    ├── clustering.py       # Поиск оптимального k и запуск KMeans
    ├── plots.py            # Функции построения графиков (elbow, silhouette, PCA и др.)
    ├── gigachat_api.py     # Взаимодействие с API GigaChat для описания кластеров
    └── init.py
```

## Конфигурация

* Период анализа задаётся пользователем (по умолчанию последние 15 дней).
* Максимальное значение k для анализа подбирается через слайдер (2-20).
* Параметры кластеризации: KMeans с пользовательским выбором k.

## Дальнейшее развитие

* Поддержка других алгоритмов кластеризации (DBSCAN, Agglomerative Clustering, Spectral Clustering).
* Возможность анализа нескольких токенов за одну сессию.
* Улучшение визуализаций и интерактивных дашбордов.

Также вы можете ознакомиться с приложением по [ссылке](https://blockchain-clustering.streamlit.app/).