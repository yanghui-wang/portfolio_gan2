#!/usr/bin/env python
"""Quick test of news_aware model_type integration."""

import logging
from pathlib import Path
from copy import deepcopy

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get repo root
repo_root = Path(__file__).parent
print(f"Repository root: {repo_root}")

# Test 1: Import the evaluator module
logger.info("Test 1: Importing evaluator module...")
try:
    from src.evaluation.evaluator import (
        _build_evaluation_cfg,
        DEFAULT_EVALUATION_CONFIG,
    )
    logger.info("✓ Evaluator imports successful")
except Exception as exc:
    logger.error("✗ Failed to import evaluator: %s", str(exc))
    exit(1)

# Test 2: Check DEFAULT_EVALUATION_CONFIG structure
logger.info("\nTest 2: Checking DEFAULT_EVALUATION_CONFIG structure...")
try:
    assert "model_type" in DEFAULT_EVALUATION_CONFIG, "model_type missing from config"
    assert DEFAULT_EVALUATION_CONFIG["model_type"] == "carhart", "Default model_type should be 'carhart'"
    assert "factor_columns" in DEFAULT_EVALUATION_CONFIG, "factor_columns missing from config"
    assert len(DEFAULT_EVALUATION_CONFIG["factor_columns"]) == 4, "Default should have 4 Carhart factors"
    logger.info("✓ DEFAULT_EVALUATION_CONFIG structure OK")
    logger.info(f"  - Default model_type: {DEFAULT_EVALUATION_CONFIG['model_type']}")
    logger.info(f"  - Default factors: {DEFAULT_EVALUATION_CONFIG['factor_columns']}")
except AssertionError as exc:
    logger.error("✗ Config structure check failed: %s", str(exc))
    exit(1)

# Test 3: Build config with model_type='carhart' (default)
logger.info("\nTest 3: Building config with model_type='carhart'...")
try:
    cfg_carhart = _build_evaluation_cfg({"model_type": "carhart"})
    assert cfg_carhart["model_type"] == "carhart", "Failed to set model_type='carhart'"
    assert len(cfg_carhart["factor_columns"]) == 4, "Carhart model should have 4 factors"
    assert cfg_carhart["factor_columns"] == ["market_beta", "SMB", "HML", "UMD"], \
        f"Unexpected factor columns: {cfg_carhart['factor_columns']}"
    logger.info("✓ Carhart model config OK")
    logger.info(f"  - Factors: {cfg_carhart['factor_columns']}")
except Exception as exc:
    logger.error("✗ Carhart config building failed: %s", str(exc))
    exit(1)

# Test 4: Build config with model_type='news_aware'
logger.info("\nTest 4: Building config with model_type='news_aware'...")
try:
    cfg_news = _build_evaluation_cfg({"model_type": "news_aware"})
    assert cfg_news["model_type"] == "news_aware", "Failed to set model_type='news_aware'"
    assert len(cfg_news["factor_columns"]) == 9, f"News-aware model should have 9 factors, got {len(cfg_news['factor_columns'])}"
    
    expected_factors = [
        "market_beta", "SMB", "HML", "UMD",  # Carhart
        "ortho_sentiment", "ortho_risk", "ortho_uncertainty",
        "ortho_macro_credit_pressure", "ortho_corporate_market_activity"  # News
    ]
    assert cfg_news["factor_columns"] == expected_factors, \
        f"Unexpected factor columns:\n  Got: {cfg_news['factor_columns']}\n  Expected: {expected_factors}"
    
    logger.info("✓ News-aware model config OK")
    logger.info(f"  - Factors ({len(cfg_news['factor_columns'])}): {cfg_news['factor_columns']}")
except Exception as exc:
    logger.error("✗ News-aware config building failed: %s", str(exc))
    exit(1)

# Test 5: Verify factor_exposures_news_aware module imports
logger.info("\nTest 5: Importing factor_exposures_news_aware module...")
try:
    from src.evaluation.factor_exposures_news_aware import (
        CARHART_FACTORS,
        NEWS_FACTORS,
        ALL_FACTORS,
    )
    assert len(CARHART_FACTORS) == 4, "CARHART_FACTORS should have 4 elements"
    assert len(NEWS_FACTORS) == 5, "NEWS_FACTORS should have 5 elements"
    assert len(ALL_FACTORS) == 9, "ALL_FACTORS should have 9 elements"
    logger.info("✓ factor_exposures_news_aware imports successful")
    logger.info(f"  - CARHART_FACTORS: {CARHART_FACTORS}")
    logger.info(f"  - NEWS_FACTORS: {NEWS_FACTORS}")
except Exception as exc:
    logger.error("✗ factor_exposures_news_aware import failed: %s", str(exc))
    exit(1)

# Test 6: Verify news_factors path is set in config
logger.info("\nTest 6: Checking news_factors path in config...")
try:
    assert "news_factors" in DEFAULT_EVALUATION_CONFIG["inputs"], \
        "news_factors missing from inputs"
    news_path = DEFAULT_EVALUATION_CONFIG["inputs"]["news_factors"]
    assert news_path, "news_factors path should not be empty"
    logger.info("✓ News factors path configured")
    logger.info(f"  - Path: {news_path}")
except AssertionError as exc:
    logger.error("✗ News factors path check failed: %s", str(exc))
    exit(1)

# Summary
logger.info("\n" + "="*60)
logger.info("✓ ALL TESTS PASSED")
logger.info("="*60)
logger.info("\nIntegration Summary:")
logger.info("  • model_type dispatch routing working")
logger.info("  • factor_columns dynamic assignment working")
logger.info("  • Carhart 4-factor support: ✓")
logger.info("  • News-aware 9-factor support: ✓")
logger.info("  • Backward compatibility (default='carhart'): ✓")
logger.info("\nReady to run: python -m src.evaluation.evaluator")
