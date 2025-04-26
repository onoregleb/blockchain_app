import streamlit as st
import numpy as np
import pandas as pd
import re

from utils.preprocessing import load_data, preprocess_data
from utils.clustering import find_optimal_clusters, perform_clustering
from utils.plots import (
    plot_elbow_method,
    plot_silhouette,
    plot_davies_bouldin,
    plot_pca_clusters
)
from utils.eda import generate_eda_plots
from utils.gigachat_api import get_ai_description_from_stats
from src.fetch_wallet import run_fetch_and_process


default_session_state = {
    'data_source': None,
    'data_loaded': False,
    'fetch_error': None,
    'fetch_warnings': None,
    'api_address_input': "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    'api_days_input': 15,
    'cluster_performed': False,
    'original_data': None,
    'processed_data': None,
    'scaled_features': None,
    'cluster_metrics': None,
    'cluster_description': None,
    'displayed_stats': None
}
for key, default in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(
    page_title="Wallet Clustering Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Анализ и кластеризация кошельков")

st.sidebar.title("Источник данных")
data_source_option = st.sidebar.radio(
    "Выберите способ получения данных:",
    ('Загрузить CSV', 'Собрать через API Etherscan'),
    index=None,
    key='data_source_choice'
)

if data_source_option == 'Загрузить CSV':
    st.session_state.data_source = 'csv'
elif data_source_option == 'Собрать через API Etherscan':
    st.session_state.data_source = 'api'

st.markdown("### 1. Загрузка или Сбор данных")

if st.session_state.data_source == 'csv':
    uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"], key="csv_uploader")
    if uploaded_file is not None and not st.session_state.data_loaded:
        try:
            with st.spinner("Загрузка и предобработка данных из CSV..."):
                data = load_data(uploaded_file)
                st.session_state.original_data = data
                scaled_features, processed_data = preprocess_data(data)
                st.session_state.scaled_features = scaled_features
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.session_state.fetch_error = None #
                st.success("Данные из CSV успешно загружены и обработаны!")
                st.rerun()
        except Exception as e:
            st.error(f"Ошибка при загрузке или обработке CSV: {str(e)}")
            st.session_state.data_loaded = False
            st.session_state.original_data = None
            st.session_state.processed_data = None
            st.session_state.scaled_features = None

elif st.session_state.data_source == 'api':
    st.subheader("Параметры для сбора данных через API")
    etherscan_api_key = st.secrets.get("ETHERSCAN_API_KEY")

    if not etherscan_api_key:
        st.warning("""
            **Ключ API Etherscan не найден!**

            Пожалуйста, убедитесь, что вы добавили строку
            `ETHERSCAN_API_KEY = "ВАШ_КЛЮЧ"`
            в ваш файл секретов `.streamlit/secrets.toml`.

            Без ключа API сбор данных через Etherscan невозможен.
        """)
        st.stop()


    # Поля ввода для API
    api_address = st.text_input(
        "Адрес контракта токена (ERC-20)",
        value=st.session_state.api_address_input,
        key="api_address"
    )
    api_days = st.number_input(
        "Количество дней для анализа (назад от текущей даты)",
        min_value=1,
        max_value=365,
        value=st.session_state.api_days_input,
        step=1,
        key="api_days"
    )

    if st.button("Начать сбор данных", key="start_api_fetch"):
        if not re.match(r'^0x[a-fA-F0-9]{40}$', api_address):
             st.error("Неверный формат адреса Ethereum. Адрес должен начинаться с '0x' и содержать 40 шестнадцатеричных символов.")
        else:
            st.session_state.api_address_input = api_address
            st.session_state.api_days_input = api_days
            st.session_state.data_loaded = False
            st.session_state.fetch_error = None
            st.session_state.fetch_warnings = None
            st.session_state.original_data = None
            st.session_state.processed_data = None
            st.session_state.scaled_features = None

            st.info(f"Запуск сбора данных для токена {api_address} за последние {api_days} дней...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(percent_complete, message):
                progress_bar.progress(percent_complete / 100.0)
                status_text.info(message)

            try:
                # Вызов функции из fetch_wallet с передачей callback
                df_result, warnings_list = run_fetch_and_process(
                    target_token_contract_address=api_address,
                    days_back=api_days,
                    api_key=etherscan_api_key,
                    progress_callback=update_progress
                )

                status_text.empty()
                progress_bar.empty()

                if df_result is not None:
                    if not df_result.empty:
                        st.success(f"Сбор данных завершен! Получено {len(df_result)} записей.")
                        st.session_state.original_data = df_result
                        st.session_state.fetch_warnings = warnings_list

                        with st.spinner("Предобработка собранных данных..."):
                            scaled_features, processed_data = preprocess_data(df_result)
                            st.session_state.scaled_features = scaled_features
                            st.session_state.processed_data = processed_data
                            st.session_state.data_loaded = True
                        st.success("Предобработка данных завершена.")
                        st.rerun()

                    else:
                        st.warning("Сбор данных завершен, но не найдено кошельков или транзакций для анализа за указанный период.")
                        st.session_state.data_loaded = False # Данных нет
                else:
                    st.error("Произошла критическая ошибка во время сбора данных. Проверьте консоль или логи для деталей.")
                    st.session_state.fetch_error = "Критическая ошибка сбора данных."
                    st.session_state.data_loaded = False

            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Произошла ошибка во время выполнения сбора или обработки данных: {str(e)}")
                st.session_state.fetch_error = str(e)
                st.session_state.data_loaded = False

elif not st.session_state.data_source:
    st.info("Пожалуйста, выберите источник данных в боковой панели слева.")


if st.session_state.data_loaded and st.session_state.original_data is not None:

    # Отображение предупреждений о лимите 10k, если они были при сборе через API
    if st.session_state.data_source == 'api' and st.session_state.fetch_warnings:
        st.warning("**Предупреждение о неполных данных:**")
        warning_message = "Из-за достижения лимита Etherscan в 10,000 транзакций для следующих дат, данные и результаты анализа могут быть неполными:\n"
        for dt in sorted(list(set(st.session_state.fetch_warnings))):
            warning_message += f"- {dt.strftime('%Y-%m-%d')}\n"
        st.markdown(warning_message)


    st.markdown("---")
    st.markdown("### 2. Исследовательский анализ данных (EDA)")
    data = st.session_state.original_data

    st.subheader("Первые 5 строк данных")
    st.dataframe(data.head())

    st.subheader("Основная статистика")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.dataframe(data[numeric_cols].describe())
    else:
        st.info("Нет числовых колонок для отображения статистики.")


    st.subheader("Распределения данных")
    try:
        eda_plots = generate_eda_plots(data[numeric_cols])
        st.subheader("Распределения исходных числовых данных")
        st.pyplot(eda_plots['original_plots'])
        st.subheader("Распределения после log1p-преобразования")
        st.pyplot(eda_plots['log_plots'])
    except Exception as e:
        st.error(f"Ошибка при генерации EDA графиков: {e}")
        st.info("Возможно, в данных отсутствуют необходимые числовые колонки.")


    st.markdown("---")
    st.markdown("### 3. Определение оптимального числа кластеров")

    if st.session_state.scaled_features is not None:
        max_k = st.slider("Максимальное k для анализа", 2, 20, 10, key="max_k_slider")

        if st.button("Рассчитать метрики кластеризации", key="calc_metrics_btn"):
             with st.spinner("Расчет метрик кластеризации..."):
                try:
                    metrics = find_optimal_clusters(
                        st.session_state.scaled_features,
                        max_k
                    )
                    st.session_state.cluster_metrics = metrics
                    st.success("Расчет метрик завершен.")
                except Exception as e:
                    st.error(f"Ошибка при расчете метрик: {e}")
                    st.session_state.cluster_metrics = None

        if st.session_state.cluster_metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_elbow_method(
                    st.session_state.cluster_metrics['inertia'],
                    st.session_state.cluster_metrics['K_range']
                ))
            with col2:
                st.pyplot(plot_silhouette(
                    st.session_state.cluster_metrics['silhouette'],
                    st.session_state.cluster_metrics['K_range']
                ))
            st.pyplot(plot_davies_bouldin(
                st.session_state.cluster_metrics['davies_bouldin'],
                st.session_state.cluster_metrics['K_range']
            ))

            # Рекомендации по k
            try:
                k_range = st.session_state.cluster_metrics['K_range']
                inertia = st.session_state.cluster_metrics['inertia']
                silhouette = st.session_state.cluster_metrics['silhouette']
                davies_bouldin = st.session_state.cluster_metrics['davies_bouldin']

                elbow_k_index = -1
                if len(inertia) >= 3:
                     diff2 = np.diff(inertia, 2)
                     elbow_k_index = np.argmax(diff2) + 2
                else:
                     elbow_k_index = 0

                silhouette_k_index = np.argmax(silhouette)
                db_k_index = np.argmin(davies_bouldin)

                st.subheader("Рекомендуемые значения k:")
                st.info(f"""
                    - Метод локтя: **{k_range[elbow_k_index]}** (наибольший изгиб инерции)
                    - Silhouette Score: **{k_range[silhouette_k_index]}** (максимальный скор)
                    - Davies-Bouldin: **{k_range[db_k_index]}** (минимальный индекс)
                """)
                st.markdown("""
                       **Примечание:** Это рекомендации. Выберите 'k', которое наилучшим образом
                       соответствует вашим целям анализа и интерпретируемости кластеров.
                       """)
            except Exception as e:
                 st.warning(f"Не удалось рассчитать рекомендуемые k: {e}")


            recommended_k_default = 4
            if st.session_state.cluster_metrics:
                 try:
                     recommended_k_default = k_range[silhouette_k_index]
                 except: pass

            selected_k = st.number_input(
                "Выберите количество кластеров (k) для финальной модели",
                min_value=2,
                max_value=max_k,
                value=int(recommended_k_default), # Преобразуем в int на всякий случай
                step=1,
                key="selected_k_input"
            )

            if st.button("Запустить кластеризацию", key="run_clustering_btn"):
                 with st.spinner(f"Выполнение KMeans с k={selected_k}..."):
                    try:
                        labels_zero_based = perform_clustering(
                            st.session_state.scaled_features,
                            selected_k
                        )

                        labels_one_based = labels_zero_based + 1

                        processed_data_copy = st.session_state.processed_data.copy()
                        processed_data_copy['cluster'] = labels_one_based
                        st.session_state.processed_data = processed_data_copy

                        if len(st.session_state.original_data) == len(labels_zero_based):
                             original_data_copy = st.session_state.original_data.copy()
                             original_data_copy['cluster'] = labels_one_based
                             st.session_state.original_data = original_data_copy
                             st.session_state.cluster_performed = True
                             st.success(f"Кластеризация завершена! Найдено кластеров: {selected_k}")
                             st.session_state.cluster_description = None
                             st.session_state.displayed_stats = None
                             st.rerun()
                        else:
                            st.error("Ошибка: Несовпадение количества строк между оригинальными данными и результатами кластеризации. Не удалось добавить метки.")
                            st.session_state.cluster_performed = False

                    except Exception as e:
                        st.error(f"Ошибка при выполнении кластеризации: {e}")
                        st.session_state.cluster_performed = False

    else:
         st.info("Данные еще не загружены или не обработаны. Загрузите CSV или соберите данные через API.")


    if st.session_state.cluster_performed and 'cluster' in st.session_state.original_data.columns:
        st.markdown("---")
        st.markdown("### 4. Результаты кластеризации")

        st.subheader("Визуализация кластеров (PCA)")
        try:
            st.pyplot(plot_pca_clusters(
                st.session_state.scaled_features,
                st.session_state.original_data['cluster'] # Используем метки из original_data
            ))
        except Exception as e:
            st.error(f"Ошибка при построении PCA графика: {e}")

        st.subheader("Статистика по кластерам (на основе оригинальных данных)")
        try:
            original_numeric_cols = st.session_state.original_data.select_dtypes(include=np.number).columns.tolist()
            if 'cluster' in original_numeric_cols:
                original_numeric_cols.remove('cluster')

            if original_numeric_cols:
                 stats = st.session_state.original_data.groupby('cluster')[original_numeric_cols].describe()

                 stats_to_display = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                 filtered_stats = stats.loc[:, pd.IndexSlice[:, stats_to_display]]

                 st.session_state.displayed_stats = filtered_stats # Сохраняем для GigaChat
                 st.dataframe(st.session_state.displayed_stats)
            else:
                 st.info("В оригинальных данных нет числовых колонок для расчета статистики по кластерам.")
                 st.session_state.displayed_stats = None


        except Exception as e:
            st.error(f"Ошибка при расчете статистики по кластерам: {e}")
            st.session_state.displayed_stats = None

        st.subheader("Распределение записей по кластерам")
        st.bar_chart(st.session_state.original_data['cluster'].value_counts())

        st.markdown("---")
        st.markdown("### 5. Описание кластеров с помощью AI (GigaChat)")

        if st.session_state.displayed_stats is not None and not st.session_state.displayed_stats.empty:
            ai_stats = None
            try:
                stats_for_ai = ['mean', 'std', 'min', 'max']
                ai_stats = st.session_state.displayed_stats.loc[:, pd.IndexSlice[:, stats_for_ai]]
                stats_markdown_text = ai_stats.to_markdown() # Конвертируем в Markdown
            except Exception as e:
                st.error(f"Ошибка при подготовке статистики для AI: {str(e)}")
                ai_stats = None

            if ai_stats is not None:
                if st.button("Получить описание кластеров от GigaChat", key="get_gigachat_desc"):
                    # Получение ключа GigaChat из секретов
                    auth_basic_value = st.secrets.get('GIGACHAT_AUTH_BASIC_VALUE')

                    if auth_basic_value:
                        with st.spinner("GigaChat анализирует статистику кластеров..."):
                            description = None
                            try:
                                # Вызов функции API GigaChat
                                description = get_ai_description_from_stats(
                                    auth_basic_value=auth_basic_value,
                                    stats_text=stats_markdown_text
                                )

                                if description:
                                    st.session_state.cluster_description = description
                                    st.success("Описание от GigaChat получено!")
                                else:
                                    st.error("Не удалось получить текст описания от GigaChat. API вернул пустой ответ.")
                                    st.session_state.cluster_description = None

                            except Exception as e:
                                st.error(f"Ошибка при взаимодействии с GigaChat API: {e}")
                                st.session_state.cluster_description = None

                    else:
                        st.warning(
                            "Для получения описания от GigaChat необходимо добавить `GIGACHAT_AUTH_BASIC_VALUE` (base64 строка Client ID:Client Secret) в секреты Streamlit (`.streamlit/secrets.toml`)."
                        )
            else:
                st.warning("Не удалось подготовить статистику для передачи в GigaChat.")

        elif st.session_state.cluster_performed:
            st.warning("Статистика по кластерам не была рассчитана или пуста. Невозможно получить описание от AI.")

        if st.session_state.cluster_description:
            st.markdown("#### Описание кластеров (сгенерировано GigaChat):")
            st.markdown(st.session_state.cluster_description)


    st.markdown("---")
    if st.button("Сбросить и начать заново", key="reset_all"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
             del st.session_state[key]
        st.rerun()

# Сообщение, если данные еще не загружены/собраны в основной части
elif st.session_state.data_source == 'api' and not st.session_state.data_loaded and not st.session_state.fetch_error:
    st.info("Введите параметры и нажмите 'Начать сбор данных' для запуска анализа через API.")
elif st.session_state.fetch_error:
    st.error(f"Произошла ошибка при последней попытке загрузки/сбора данных: {st.session_state.fetch_error}")
    st.info("Исправьте ошибку (например, проверьте API ключ или формат файла) и попробуйте снова.")