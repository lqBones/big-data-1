# big_data_example.py

import dask.dataframe as dd
import numpy as np

# 1. Генерация "больших данных" — 100 млн строк (имитация логов/транзакций)
# В реальности данные поступают из внешних источников (логи, датчики, БД и т.д.)
print("Генерация больших данных...")

# Создаём Dask DataFrame "лениво" — без загрузки в память
df = dd.from_pandas(
    pd.DataFrame({
        'user_id': np.random.randint(1, 1_000_000, size=100_000_000),
        'amount': np.random.uniform(0.1, 1000.0, size=100_000_000),
        'category': np.random.choice(['food', 'tech', 'travel', 'entertainment'], size=100_000_000)
    }),
    npartitions=100  # разбиваем на 100 частей для параллельной обработки
)

# Сохраняем в файл (как этап "приёма и хранения")
print("Сохранение данных в CSV...")
df.to_csv('big_data_partition_*.csv', index=False)

# 2. Чтение и обработка (пакетная аналитика)
print("Чтение и агрегация данных...")
df_loaded = dd.read_csv('big_data_partition_*.csv')

# Агрегация: сумма и среднее по категориям
result = df_loaded.groupby('category').agg({
    'amount': ['sum', 'mean'],
    'user_id': 'count'
}).compute()  # запуск вычислений

# 3. Вывод результата (аналитика + представление)
print("\nРезультат анализа больших данных:")
print(result)

# Сохраняем результат
result.to_json('big_data_analysis.json')
print("\nАнализ завершён. Результат сохранён в 'big_data_analysis.json'")