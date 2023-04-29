##  Прогноз уровня кредитных потерь в банковской системе России

### Описание репозитория:

#### 1. Data

- Включает в себя данные для обучения моделей и прогнозирования;
- В качестве данных для обучения используется информация, собранная с различных источников (Росстат, ЕМИСС, Банк России...);
- В качестве данных для прогнозирования используется базовый вариант развития экономики РФ (Банк России).

#### 2. Lib

Включает в себя используемые в процессе моделирования функции, а также список факторов для итоговой модели.

#### 3. Experiments

- Включает в себя скрипты и файлы, которые использовались для выбора архитектуры модели, отбора признаков и построения сценариев;
- Скрипты "linear_regression.ipynb" и "neural_net.ipynb" используются для выбора архитектуры итоговой модели. Результаты прогона этих скриптов накапливаются в excel-файлах, лежащих в данном репозитории;
- Скрипт "feature_selection.ipynb" производит отбор признаков с помощью "жадного отбора" и регуляризации;
- Скрипт "var.ipynb" необходим для определения критических значений приростов макрофакторов, чтобы строить сценарии их изменения в случае шоков.

#### 4. Models

- Включает в себя скрипты по обучению и использованию моделей;
- Скрипт "train_models.ipynb" обучает линейную регрессию на итоговом наборе признаков, полученных в ходе эксперимента;
- Скрипт "predict_models.ipynb" строит поквартальные прогнозы.

#### 5. Results

- Включает в себя основные результаты моделирования
- Файл "Linreg_Results.xlsx" содержит информацию о коэффициентах и качестве модели, а также квартальные и годовые прогнозы по итоговой модели на кросс-валидации;
- Файл "Quarter_Predictions.xlsx" содержит информацию о квартальных прогнозах, построенных в скрипте "predict_models.ipynb";
- Файл "VaR.xlsx" содержит информацию о критических значениях приростов макрофакторов.

#### 6. Old

Включает в себя все старые наработки.



