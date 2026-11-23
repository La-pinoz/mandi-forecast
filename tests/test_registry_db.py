"""model_registry: storing a LightGBM model and loading the champion (needs TEST_DATABASE_URL)."""

import math
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import psycopg
import pytest

from mandi.features.build import build_feature_table
from mandi.models.lgbm import FEATURES_PER_HORIZON, predict_prices
from mandi.models.registry import load_champion, register_model
from mandi.store.runs import start_run

IST = timezone(timedelta(hours=5, minutes=30))
TRAINED_AT = datetime(2026, 11, 23, 9, 0, tzinfo=IST)


def register(conn: psycopg.Connection, models, model_version: str, promote: bool) -> None:
    run_id = start_run(conn, "test")
    register_model(
        conn,
        model_version,
        models,
        params={"layout": "per_horizon"},
        backtest_metrics={"lgbm.v1": {"n": 10, "MASE": 0.9}},
        training_data_as_of=TRAINED_AT,
        training_rows=100,
        run_id=run_id,
        promote=promote,
    )
    conn.commit()


def test_the_loaded_champion_predicts_exactly_like_the_trained_models(schema_conn, small_lgbm_models):
    register(schema_conn, small_lgbm_models, "lgbm.v1", promote=True)

    champion = load_champion(schema_conn)

    assert champion.model_version == "lgbm.v1"
    assert champion.features == FEATURES_PER_HORIZON
    assert sorted(champion.models) == [1, 2, 3, 4, 5, 6, 7]
    # Prices the models never saw, at a different level, so the trees and categories must both round-trip.
    series_info = pd.DataFrame({"series_id": [1], "market": ["Barasat"], "crop": ["Tomato"]})
    prices = {}
    spreads = {}
    for offset in range(40):
        day = date(2026, 10, 1) + timedelta(days=offset)
        prices[day] = 3000 + 25 * math.sin(offset)
        spreads[day] = 0.15
    table = build_feature_table({1: prices}, {1: spreads}, series_info, [date(2026, 11, 10)])
    loaded_forecasts = predict_prices(champion.models, table).tolist()
    assert loaded_forecasts == predict_prices(small_lgbm_models, table).tolist()


def test_the_registry_row_keeps_what_the_model_was_trained_with(schema_conn, small_lgbm_models):
    register(schema_conn, small_lgbm_models, "lgbm.v1", promote=True)

    row = schema_conn.execute(
        "SELECT params, features, backtest_metrics, training_data_as_of, training_rows, "
        "promoted_at IS NOT NULL "
        "FROM model_registry"
    ).fetchone()
    assert row == (
        {"layout": "per_horizon"},
        FEATURES_PER_HORIZON,
        {"lgbm.v1": {"n": 10, "MASE": 0.9}},
        TRAINED_AT,
        100,
        True,
    )


def test_no_champion_is_an_error(schema_conn, small_lgbm_models):
    with pytest.raises(RuntimeError, match="no champion"):
        load_champion(schema_conn)

    register(schema_conn, small_lgbm_models, "lgbm.v1", promote=False)  # registered, not promoted
    with pytest.raises(RuntimeError, match="no champion"):
        load_champion(schema_conn)


def test_a_second_champion_cannot_be_promoted_while_one_is_active(schema_conn, small_lgbm_models):
    register(schema_conn, small_lgbm_models, "lgbm.v1", promote=True)

    with pytest.raises(psycopg.errors.UniqueViolation):
        register(schema_conn, small_lgbm_models, "lgbm.v2", promote=True)
    schema_conn.rollback()

    assert load_champion(schema_conn).model_version == "lgbm.v1"
