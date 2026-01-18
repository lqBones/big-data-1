# big_data_pipeline.py
import os
import logging
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Параметры
DATA_DIR = Path("data")
RAW_DATA_PATTERN = DATA_DIR / "raw" / "transactions_*.parquet"
ANALYTICS_OUTPUT = DATA_DIR / "analytics" / "summary.json"
N_ROWS = 100_000_000  # 100 млн строк
N_PARTITIONS = 200    # Больше партиций = лучше параллелизм
CATEGORIES = ["food", "tech", "travel", "entertainment"]

def generate_large_dataset():
    """Генерация синтетического набора данных (Big Data) и сохранение в Parquet."""
    logger.info("Начало генерации синтетических данных...")
    
    # Создаём директории
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "analytics").mkdir(parents=True, exist_ok=True)

    # Генерация "ленивого" Dask DataFrame без загрузки в память
    df = dd.from_map(
        lambda i: pd.DataFrame({
            'user_id': np.random.randint(1, 2_000_000, size=N_ROWS // N_PARTITIONS),
            'amount': np.random.exponential(scale=50.0, size=N_ROWS // N_PARTITIONS),  # Более реалистичное распределение
            'category': np.random.choice(CATEGORIES, size=N_ROWS // N_PARTITIONS, p=[0.4, 0.2, 0.25, 0.15])
        }),
        range(N_PARTITIONS),
        divisions=[None] * (N_PARTITIONS + 1)
    )

    # Сохраняем в Parquet — эффективный формат для Big Data
    output_path = str(DATA_DIR / "raw" / "transactions_*.parquet")
    logger.info(f"Сохранение данных в Parquet: {output_path}")
    df.to_parquet(output_path, compression="snappy", write_index=False)
    logger.info("Генерация завершена.")

def analyze_data():
    """Анализ больших данных: агрегация, статистика, сохранение результата."""
    logger.info("Загрузка данных из Parquet...")
    df = dd.read_parquet(RAW_DATA_PATTERN)

    logger.info("Выполнение агрегации...")
    summary = (
        df.groupby("category")
        .agg(
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            transaction_count=("amount", "count"),
            unique_users=("user_id", "nunique")
        )
        .round(2)
        .compute()  # Запуск вычислений
    )

    logger.info("Сохранение аналитики...")
    summary.to_json(ANALYTICS_OUTPUT, orient="index", indent=2)
    logger.info(f"Анализ завершён. Результат сохранён в {ANALYTICS_OUTPUT}")

    # Вывод в консоль
    print("\n📊 Сводка по категориям:")
    print(summary)

if __name__ == "__main__":
    # Шаг 1: Генерация данных (выполняется один раз)
    if not list((DATA_DIR / "raw").glob("*.parquet")):
        generate_large_dataset()
    
    # Шаг 2: Анализ
    analyze_data()