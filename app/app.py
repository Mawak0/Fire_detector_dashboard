import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_model(model_name):
    if model_name == "CatBoost":
        from_file = CatBoostClassifier()
        return from_file.load_model("models/cat_boost.cbm")
    with open(model_name, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data(path_to_csv):
    return pd.read_csv(path_to_csv, index_col=0)


st.set_page_config(page_title="Fire Alarm ML Dashboard", layout="wide")
menu = st.sidebar.radio("Навигация", ["О разработчике", "О датасете", "Визуализации", "Предсказание"])

if menu == "О разработчике":
    st.title("Разработчик проекта")
    st.subheader("Дрожжачих Артём Дмитриевич")
    st.text("Группа: ФИТ-231")
    st.markdown("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

elif menu == "О датасете":
    st.title("Описание датасета")
    st.markdown("""
    **Область:** Предсказание срабатывания пожарной тревоги на основе данных с сенсоров воздуха.

    **Целевая переменная:** `Fire Alarm` (1 — тревога, 0 — нет тревоги)

    **Признаки:**
    - `Temperature[C]`: температура воздуха  
    - `Humidity[%]`: влажность  
    - `TVOC[ppb]`: летучие органические соединения  
    - `eCO2[ppm]`: эквивалент CO₂  
    - `Raw H2`: молекулярный водород  
    - `Raw Ethanol`: этанол  
    - `Pressure[hPa]`: атмосферное давление  
    - `PM1.0`, `PM2.5`: пылевые частицы  
    - `NC0.5`, `NC1.0`, `NC2.5`: количество частиц  
    - `CNT`: количество измерений
    - `Day`: День месяца
    - `Hour`: Час в 24 часовом формате
    """)
    st.subheader("Предобработка данных")
    st.markdown("""
    - В данных было обнаружено небольшое количество пропущенных значений.
    - Пропущенные значения заменены медианными значениями соответствующих столбцов.
    - Проверено отсутствие дубликатов – дубликаты отсутствуют.
    - Столбец Fire Alarm преобразован из типа str в bool.
    - Столбец UTC преобразован в Datetime, дополнительно выделены день и час.
    """)


elif menu == "Визуализации":
    df = load_data("data/smoke_detector_task_preprocessed.csv")
    st.title("Анализ данных")

    st.subheader("1. Распределение температуры")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Temperature[C]'], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("2. Корреляционная матрица")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("3. Boxplot: eCO2 по тревоге")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=df, x='Fire Alarm', y='eCO2[ppm]', ax=ax3)
    st.pyplot(fig3)

    st.subheader("4. Взаимосвязь температуры и влажности")
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x='Temperature[C]', y='Humidity[%]', hue='Fire Alarm', ax=ax4)
    st.pyplot(fig4)


elif menu == "Предсказание":
    st.title("Предсказание срабатывания пожарной тревоги")

    mode = st.radio("Выберите режим:", ["Ручной ввод", "Загрузка CSV"])

    # Список всех доступных моделей
    all_models = ["KNN", "Gradient Boosting", "CatBoost", "Bagging", "Stacking", "MLPClassifier"]
    # Теперь — мультиселект
    selected_models = st.multiselect("Модели:", all_models)

    # Функция для загрузки одной модели по её названию
    @st.cache_resource
    def load_selected_model(name):
        if name == "KNN":
            return load_model("models/knn_model.pkl")
        elif name == "Gradient Boosting":
            return load_model("models/GradientBoostingClassifier_model.pkl")
        elif name == "CatBoost":
            return load_model("CatBoost")
        elif name == "Bagging":
            return load_model("models/BaggingClassifier_model.pkl")
        elif name == "Stacking":
            return load_model("models/StackingClassifier_model.pkl")
        elif name == "MLPClassifier":
            return load_model("models/MLP_classifier.pkl")
        else:
            return None  # на случай передачи неизвестного имени

    # Загружаем выбранные модели в словарь
    models_dict = {}
    for name in selected_models:
        model_obj = load_selected_model(name)
        if model_obj is not None:
            models_dict[name] = model_obj

    # Режим «Ручной ввод»
    if mode == "Ручной ввод":
        st.subheader("Введите данные")

        # Поля ввода для каждого признака
        values = {}
        values['Temperature[C]'] = st.number_input("Температура (°C)")
        values['Humidity[%]']    = st.number_input("Влажность (%)")
        values['TVOC[ppb]']      = st.number_input("TVOC (ppb)")
        values['eCO2[ppm]']      = st.number_input("eCO₂ (ppm)")
        values['Raw H2']         = st.number_input("Raw H2")
        values['Raw Ethanol']    = st.number_input("Raw Ethanol")
        values['Pressure[hPa]']  = st.number_input("Давление (hPa)")
        values['PM1.0']          = st.number_input("PM1.0")
        values['PM2.5']          = st.number_input("PM2.5")
        values['NC0.5']          = st.number_input("NC0.5")
        values['NC1.0']          = st.number_input("NC1.0")
        values['NC2.5']          = st.number_input("NC2.5")
        values['CNT']            = st.number_input("CNT")
        values['Day']            = st.number_input("Day", format="%d", step=1, min_value=1, max_value=31)
        values['Hour']           = st.number_input("Hour", format="%d", step=1, min_value=0, max_value=23)

        # Собираем в DataFrame
        input_df = pd.DataFrame([values])

        if selected_models:
            st.subheader("Результаты предсказания")
            for name, model in models_dict.items():
                pred = model.predict(input_df)[0]
                result_text = "🔥 Пожарная тревога!" if pred == 1 else "✅ Всё спокойно."
                st.write(f"**{name}**: {result_text}")
        else:
            st.warning("Пожалуйста, выберите хотя бы одну модель для предсказания.")

    # Режим «Загрузка CSV»
    else:
        st.subheader("Загрузите CSV-файл")
        uploaded_file = st.file_uploader("Файл должен содержать все необходимые признаки", type=["csv"])
        if uploaded_file:
            try:
                test_df = pd.read_csv(uploaded_file)
                if selected_models:
                    st.subheader("Результаты предсказания")
                    for name, model in models_dict.items():
                        preds = model.predict(test_df)
                        test_df[f"Prediction_{name.replace(' ', '_')}"] = preds
                    st.write(test_df.head())
                    st.success("Предсказование выполнено успешно!")
                else:
                    st.warning("Пожалуйста, выберите хотя бы одну модель для предсказания.")
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")

