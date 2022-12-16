Spark Pipeline - реализация конвейера обработки данных несколькими моделями на Pyspark

pipeline = Pipeline(stages=[assemble, scale, KMeans_algo, rf, column_dropper, lr])

1 step: assemble - преобразование вектора признаков в векторный формат

2 step: scale - стандартизация вектора признаков

3 step: KMeans_algo - применение алгоритма К-средних и запись меток кластеров в столбец clust_preds (первая модель)

4 step: rf - обучение алгоритма RandomForestClassifier на метках кластеризации и запись меток классов в столбец class_preds (вторая модель)

5 step: column_dropper - объект класса Transformer для удаления служебных столбцов из исходного датафрейма перед применением третьей модели для предотвращения конфликта имен

6 step: lr - обучение модели Logistic Regression на метках кластеров и получение распределения вероятностей на метках по каждому вектору признаков. В итоговый столбец lr_estim_rf записывается вероятность, которую присвоила модель Logistic Regression меткам классов модели Random Forest, то есть оценка третьей моделью результатов раюоты второй модели.

Итоговый датасет может быть сохранен в csv формате, куда включаются код и наименование продукта, столбцы признаков и столбцы меток кластеров, классов от RF и оценка меток от RF моделью LR.
