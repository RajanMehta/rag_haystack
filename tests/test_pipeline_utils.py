import os

from haystack_api.pipeline.pipeline_configs import pipeline_utils


def test_get_model_path_returns_bare_name_when_local_dir_unset(monkeypatch, caplog):
    monkeypatch.setattr(pipeline_utils, "LOCAL_MODEL_DIR", None)
    caplog.set_level("INFO", logger=pipeline_utils.logger.name)

    assert pipeline_utils.get_model_path("sentence-transformers/all-mpnet-base-v2") == (
        "sentence-transformers/all-mpnet-base-v2"
    )
    assert any("LOCAL_MODEL_DIR not set" in r.message for r in caplog.records)


def test_get_model_path_returns_local_path_when_dir_exists(monkeypatch, tmp_path, caplog):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    (tmp_path / model_name).mkdir(parents=True)
    monkeypatch.setattr(pipeline_utils, "LOCAL_MODEL_DIR", str(tmp_path))
    caplog.set_level("INFO", logger=pipeline_utils.logger.name)

    result = pipeline_utils.get_model_path(model_name)

    assert result == os.path.join(str(tmp_path), model_name)
    assert any("Loading" in r.message and model_name in r.message for r in caplog.records)


def test_get_model_path_warns_and_falls_back_when_dir_missing(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(pipeline_utils, "LOCAL_MODEL_DIR", str(tmp_path))
    caplog.set_level("WARNING", logger=pipeline_utils.logger.name)

    result = pipeline_utils.get_model_path("sentence-transformers/all-mpnet-base-v2")

    assert result == "sentence-transformers/all-mpnet-base-v2"
    assert any("falling back to HuggingFace" in r.message for r in caplog.records)
