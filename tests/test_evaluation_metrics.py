from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.evaluator import run_evaluation
from src.evaluation.factor_exposures import build_carhart_factor_exposures
from src.evaluation.metrics_behavior import compute_factor_exposures, compute_strategy_stability
from src.evaluation.metrics_portfolio import (
    concentration_error,
    count_error,
    herfindahl_index,
    holding_count,
    portfolio_turnover,
    turnover_error,
)
from src.features.tensor_builder import DatasetBundle, RealPortfolioDataset
from src.training.evaluation_exporter import export_evaluation_artifacts


def test_portfolio_metrics_on_sparse_vectors() -> None:
    true = pd.Series({"A": 0.6, "B": 0.4})
    pred = pd.Series({"A": 0.5, "C": 0.5})
    prev = pd.Series({"A": 0.3, "B": 0.7})

    assert holding_count(pred, 0.01) == 2
    assert count_error(true, pred, 0.01) == 0.0
    assert np.isclose(herfindahl_index(true), 0.52)
    assert np.isclose(concentration_error(true, pred), 0.02)
    assert np.isclose(portfolio_turnover(true, prev), 0.6)
    assert np.isclose(turnover_error(true, pred, prev), 0.8)


def test_factor_exposure_shape_and_values() -> None:
    portfolios = pd.DataFrame(
        {
            "fund_id": ["F1", "F1"],
            "date": ["2020-01-31", "2020-01-31"],
            "asset_id": ["A", "B"],
            "w_pred": [0.25, 0.75],
            "market_beta": [1.0, 2.0],
            "SMB": [0.5, -0.5],
        }
    )

    exposures = compute_factor_exposures(
        portfolios,
        weight_col="w_pred",
        factor_columns=["market_beta", "SMB"],
        columns={},
        normalize_weights=False,
    )

    assert exposures.shape[0] == 1
    assert np.isclose(exposures.loc[0, "market_beta"], 1.75)
    assert np.isclose(exposures.loc[0, "SMB"], -0.25)


def test_strategy_stability_handcrafted_two_period_example() -> None:
    portfolios = pd.DataFrame(
        {
            "fund_id": ["A", "B", "A", "B"],
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "asset_id": ["S1", "S1", "S1", "S1"],
            "w_pred": [1.0, 1.0, 1.0, 1.0],
            "market_beta": [1.0, 3.0, 4.0, 4.0],
        }
    )

    by_fund, summary = compute_strategy_stability(
        portfolios,
        factor_columns=["market_beta"],
        columns={},
        normalize_weights=False,
        weight_columns=["w_pred"],
    )

    assert by_fund.shape[0] == 2
    assert np.allclose(by_fund["factor_tilt_stability"], [1.0, 1.0])
    assert np.isclose(summary.loc[0, "mean"], 1.0)


def test_carhart_factor_exposure_estimation_tiny_panel() -> None:
    stock_returns = pd.DataFrame(
        {
            "permno": [1, 1, 1, 1],
            "date": ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"],
            "ret": [0.01, 0.03, 0.02, 0.04],
        }
    )
    factors = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"],
            "mktrf": [0.01, 0.02, 0.03, 0.04],
            "smb": [0.00, 0.01, 0.00, 0.01],
            "hml": [0.02, 0.01, 0.02, 0.01],
            "umd": [0.01, 0.00, 0.01, 0.00],
            "rf": [0.0, 0.0, 0.0, 0.0],
        }
    )

    exposures = build_carhart_factor_exposures(
        stock_returns,
        factors,
        cfg={"lookback_periods": 4, "min_periods": 3, "ridge_penalty": 1e-8},
    )

    assert exposures.shape[0] == 2
    assert {"asset_id", "date", "market_beta", "SMB", "HML", "UMD", "status"}.issubset(exposures.columns)
    assert np.isfinite(exposures[["market_beta", "SMB", "HML", "UMD"]].to_numpy()).all()


def test_evaluate_stage_runs_on_toy_dataset_and_records_skips(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / "artifacts/evaluation").mkdir(parents=True)
    (project_root / "raw").mkdir(parents=True)
    outputs_dir = project_root / "outputs"

    portfolio = pd.DataFrame(
        {
            "model_name": ["toy"] * 8,
            "run_id": ["run1"] * 8,
            "prediction_source": ["model"] * 8,
            "split": ["test"] * 8,
            "fund_id": ["F1", "F1", "F2", "F2", "F1", "F1", "F2", "F2"],
            "date": ["2020-01-31"] * 4 + ["2020-02-29"] * 4,
            "asset_id": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "w_true": [0.7, 0.3, 0.2, 0.8, 0.6, 0.4, 0.3, 0.7],
            "w_pred": [0.6, 0.4, 0.4, 0.6, 0.55, 0.45, 0.25, 0.75],
            "w_prev": [0.5, 0.5, 0.5, 0.5, 0.7, 0.3, 0.2, 0.8],
            "style_label": ["growth", "growth", "value", "value", "growth", "growth", "value", "value"],
        }
    )
    portfolio.to_csv(project_root / "artifacts/evaluation/portfolio_predictions.csv", index=False)

    factors = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "asset_id": ["A", "B", "A", "B"],
            "market_beta": [1.0, 1.2, 0.9, 1.4],
            "SMB": [0.1, -0.1, 0.2, -0.2],
            "HML": [0.3, 0.1, 0.2, 0.4],
            "UMD": [0.0, 0.5, 0.1, 0.3],
        }
    )
    factors.to_csv(project_root / "raw/stock_characteristics.csv", index=False)

    returns = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "asset_id": ["A", "B", "A", "B"],
            "ret": [0.01, 0.02, -0.01, 0.03],
        }
    )
    returns.to_csv(project_root / "raw/stock_returns.csv", index=False)

    cfg = {
        "evaluation": {
            "model_name": "toy",
            "run_id": "run1",
            "frontier": {
                "num_random_portfolios": 25,
                "num_random_reference_portfolios": 5,
                "lookback_periods": 2,
                "min_periods": 1,
            },
            "inputs": {
                "portfolio_predictions": "artifacts/evaluation/portfolio_predictions.parquet",
                "representation_embeddings": "artifacts/embeddings/missing_embeddings.parquet",
                "counterfactual_transfers": "artifacts/evaluation/missing_counterfactual.parquet",
            },
        }
    }

    run_evaluation(project_root, cfg, outputs_dir)

    eval_dir = outputs_dir / "evaluation"
    assert (eval_dir / "portfolio_metrics_by_sample.parquet").exists()
    assert (eval_dir / "portfolio_metrics_summary.csv").exists()
    assert (eval_dir / "stability_by_fund.parquet").exists()
    assert (eval_dir / "frontier_metrics_by_sample.parquet").exists()
    assert (eval_dir / "evaluation_report.md").exists()

    summary = pd.read_csv(eval_dir / "portfolio_metrics_summary.csv")
    assert set(summary["metric"]) == {"L_count", "L_concentration", "L_turnover"}

    skipped = json.loads((eval_dir / "skipped_metrics.json").read_text(encoding="utf-8"))
    skipped_metrics = {row["metric"] for row in skipped["skipped"]}
    assert "strategy_representation_linear_probe" in skipped_metrics
    assert "counterfactual_transfer" in skipped_metrics


def test_training_exporter_writes_evaluation_artifacts(tmp_path: Path) -> None:
    derived_dir = tmp_path / "derived"
    derived_dir.mkdir()
    (tmp_path / "raw").mkdir()
    pd.DataFrame(
        {
            "crsp_fundno": ["000105"],
            "caldt": ["2019-12-31"],
            "lipper_class": ["LCGE"],
        }
    ).to_csv(tmp_path / "raw/lipper.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "stock_id": [101, 202],
            "asset_idx": [0, 1],
            "mkt_cap": [10.0, 5.0],
        }
    ).to_parquet(derived_dir / "stock_universe_panel.parquet", index=False)

    dataset = RealPortfolioDataset(
        x=np.ones((1, 2, 1), dtype=np.float32),
        r=np.zeros((1, 2), dtype=np.float32),
        w_prev=np.array([[0.4, 0.6]], dtype=np.float32),
        w_t=np.array([[0.7, 0.3]], dtype=np.float32),
        meta=pd.DataFrame({"fund_id": [105], "date": pd.to_datetime(["2020-01-31"]), "split": ["test"]}),
    )
    empty = RealPortfolioDataset(
        x=np.zeros((0, 2, 1), dtype=np.float32),
        r=np.zeros((0, 2), dtype=np.float32),
        w_prev=np.zeros((0, 2), dtype=np.float32),
        w_t=np.zeros((0, 2), dtype=np.float32),
        meta=pd.DataFrame(columns=["fund_id", "date", "split"]),
    )
    bundle = DatasetBundle(train_dataset=empty, val_dataset=empty, test_dataset=dataset, summary=pd.DataFrame())

    class Encoder:
        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def __call__(self, x: torch.Tensor, r: torch.Tensor, w_prev: torch.Tensor, w_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ones((x.shape[0], 2), dtype=x.dtype), torch.zeros((x.shape[0], 2), dtype=x.dtype)

    class Allocator:
        def eval(self) -> None:
            pass

        def train(self) -> None:
            pass

        def __call__(self, x: torch.Tensor, r: torch.Tensor, phi: torch.Tensor, w_prev: torch.Tensor) -> torch.Tensor:
            return torch.full_like(w_prev, 0.5)

    class Logger:
        def info(self, *args: object) -> None:
            pass

        def warning(self, *args: object) -> None:
            pass

    class Trainer:
        device = torch.device("cpu")
        encoder = Encoder()
        allocator = Allocator()

    cfg = {
        "evaluation": {
            "inputs": {
                "portfolio_predictions": "artifacts/evaluation/portfolio_predictions.parquet",
                "representation_embeddings": "artifacts/embeddings/strategy_embeddings.parquet",
            },
            "export": {"splits": ["test"], "batch_size": 1},
            "labels": {
                "path": "raw/lipper.csv",
                "fund_id_column": "crsp_fundno",
                "date_column": "caldt",
                "source_label_column": "lipper_class",
                "output_column": "style_label",
                "merge_method": "asof_backward",
            },
        }
    }

    paths = export_evaluation_artifacts(
        Trainer(),
        bundle,
        project_root=tmp_path,
        derived_dir=derived_dir,
        eval_cfg=cfg,
        run_id="runx",
        logger=Logger(),
    )

    assert paths["portfolio_predictions"] == tmp_path / "artifacts/evaluation/portfolio_predictions.parquet"
    assert paths["strategy_embeddings"] == tmp_path / "artifacts/embeddings/strategy_embeddings.parquet"

    predictions = pd.read_parquet(paths["portfolio_predictions"])
    embeddings = pd.read_parquet(paths["strategy_embeddings"])
    assert predictions.shape[0] == 2
    assert predictions["asset_id"].tolist() == ["101", "202"]
    assert np.allclose(predictions["w_pred"], [0.5, 0.5])
    assert predictions["style_label"].tolist() == ["LCGE", "LCGE"]
    assert {"phi_1", "phi_2"}.issubset(embeddings.columns)
    assert embeddings.loc[0, "style_label"] == "LCGE"
