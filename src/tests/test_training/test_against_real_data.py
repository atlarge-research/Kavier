"""Manual: python src/tests/test_training/test_against_real_data.py"""

from pathlib import Path

import pandas as pd

from kavier_training.core.engine import simulate_full_training

from .conftest import simulatable_mask, throughput_column


def load_real_data() -> pd.DataFrame:
    data_path = (
        Path(__file__).resolve().parent.parent.parent / "kavier_training" / "data" / "input" / "validation_clean.csv"
    )
    if not data_path.exists():
        raise FileNotFoundError(f"Missing validation data: {data_path}")
    df = pd.read_csv(data_path)
    return df.loc[simulatable_mask(df)].head(50)


def main() -> None:
    df = load_real_data()
    tcol = throughput_column(df)
    results = []

    for row in df.itertuples(index=False):
        pred = simulate_full_training(
            model_name=row.model_name,
            method=row.method,
            gpu_model=row.gpu_model,
            tokens_per_sample=int(row.tokens_per_sample),
            batch_size=int(row.batch_size),
            number_gpus=int(row.number_gpus),
            number_nodes=int(row.number_nodes),
        )["train_tokens_per_second"]
        actual = float(getattr(row, tcol))
        mape = abs(pred - actual) / actual * 100 if actual > 0 else 0.0
        results.append({"model": row.model_name, "mape": mape})

    df_results = pd.DataFrame(results)
    print(f"Median MAPE: {df_results['mape'].median():.2f}%")
    print(f"Mean MAPE:   {df_results['mape'].mean():.2f}%")


if __name__ == "__main__":
    main()
