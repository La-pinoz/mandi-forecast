"""M09: store trained LightGBM models in `model_registry` and load the champion (blueprint 10).

A model is saved as LightGBM's own text format (`Booster.model_to_string`), one text per horizon, in the
`models` jsonb column. Loading is the reverse: `lgb.Booster(model_str=text)`. The text holds the trees and
the categories of the pandas categorical features, so the loaded model predicts exactly like the trained one.

The champion is the one row with `promoted_at` set and `retired_at` empty (a unique index allows only one).
"""

from dataclasses import dataclass
from datetime import datetime

import lightgbm as lgb
import psycopg
from psycopg.types.json import Jsonb

INSERT_MODEL = """
INSERT INTO model_registry (
    model_version, params, features, backtest_metrics, models, training_data_as_of, training_rows,
    promoted_at, run_id
)
VALUES (
    %(model_version)s, %(params)s, %(features)s, %(backtest_metrics)s, %(models)s, %(training_data_as_of)s,
    %(training_rows)s, CASE WHEN %(promote)s THEN now() END, %(run_id)s
)
"""

SELECT_CHAMPION = """
SELECT model_version, features, models
FROM model_registry
WHERE promoted_at IS NOT NULL AND retired_at IS NULL
"""


@dataclass
class Champion:
    model_version: str
    features: list[str]
    models: dict[int, lgb.Booster]  # {horizon: model}, as returned by lgbm.train


def register_model(
    conn: psycopg.Connection,
    model_version: str,
    models: dict[int, lgb.Booster],
    params: dict,
    backtest_metrics: dict,
    training_data_as_of: datetime,
    training_rows: int,
    run_id: int,
    promote: bool,
) -> None:
    """Insert one model version; with `promote`, it becomes the champion. Does not commit.

    Raises ValueError if the horizons' models don't share one feature list. Promoting while another champion
    exists fails on the unique index: retiring the old champion is a separate, explicit step (M13).
    """
    features = None
    model_texts = {}
    for horizon, model in models.items():
        if features is None:
            features = model.feature_name()
        elif model.feature_name() != features:
            raise ValueError(f"horizon {horizon}'s model has different features from the others")
        model_texts[str(horizon)] = model.model_to_string()

    conn.execute(
        INSERT_MODEL,
        {
            "model_version": model_version,
            "params": Jsonb(params),
            "features": Jsonb(features),
            "backtest_metrics": Jsonb(backtest_metrics),
            "models": Jsonb(model_texts),
            "training_data_as_of": training_data_as_of,
            "training_rows": training_rows,
            "promote": promote,
            "run_id": run_id,
        },
    )


def load_champion(conn: psycopg.Connection) -> Champion:
    """The current champion with its models rebuilt from their stored text.

    Raises RuntimeError if there is no champion, or if a stored model's features differ from the registered
    feature list (the row would not describe the model it holds).
    """
    row = conn.execute(SELECT_CHAMPION).fetchone()
    if row is None:
        raise RuntimeError(
            "model_registry has no champion. Register one first: uv run python -m mandi.models.register_lgbm"
        )
    model_version, features, model_texts = row

    models = {}
    for horizon_text, model_text in model_texts.items():
        model = lgb.Booster(model_str=model_text)
        if model.feature_name() != features:
            raise RuntimeError(
                f"{model_version}: horizon {horizon_text}'s model doesn't match its feature list"
            )
        models[int(horizon_text)] = model
    return Champion(model_version=model_version, features=features, models=models)
