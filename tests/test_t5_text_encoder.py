"""T5 prompt encoder 默认本地目录回归测试."""

from __future__ import annotations

import importlib
import sys
import types


class _FakeLoadedT5Model:
    """最小假模型,用于观察设备迁移和 eval 调用."""

    def __init__(self) -> None:
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def _import_t5_text_encoder_module(monkeypatch):
    """注入最小 transformers 假模块,避免测试依赖真实第三方包."""

    fake_transformers = types.ModuleType("transformers")

    class _FakeTransformersLogging:
        @staticmethod
        def set_verbosity_error() -> None:
            return None

    class _FakeTokenizerFast:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("test should monkeypatch tokenizer loader before use")

    class _FakeEncoderModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("test should monkeypatch model loader before use")

    fake_transformers.logging = _FakeTransformersLogging()
    fake_transformers.T5TokenizerFast = _FakeTokenizerFast
    fake_transformers.T5EncoderModel = _FakeEncoderModel

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    sys.modules.pop("cosmos_predict1.auxiliary.t5_text_encoder", None)
    return importlib.import_module("cosmos_predict1.auxiliary.t5_text_encoder")


def _import_pipeline_module(monkeypatch):
    """按同一份假依赖导入 pipeline,确保测试环境稳定."""

    t5_text_encoder_module = _import_t5_text_encoder_module(monkeypatch)
    fake_guardrail_presets = types.ModuleType("cosmos_predict1.auxiliary.guardrail.common.presets")
    fake_guardrail_presets.create_text_guardrail_runner = lambda checkpoint_dir=None: None
    fake_guardrail_presets.create_video_guardrail_runner = lambda checkpoint_dir=None: None

    monkeypatch.setitem(sys.modules, "cosmos_predict1.auxiliary.guardrail.common.presets", fake_guardrail_presets)
    sys.modules.pop("cosmos_predict1.utils.base_world_generation_pipeline", None)
    pipeline_module = importlib.import_module("cosmos_predict1.utils.base_world_generation_pipeline")
    return t5_text_encoder_module, pipeline_module


def test_cosmos_t5_text_encoder_defaults_to_fixed_local_dir(monkeypatch) -> None:
    """直接实例化编码器时,默认也必须指向固定本地目录."""

    t5_text_encoder_module = _import_t5_text_encoder_module(monkeypatch)
    captured_calls: dict[str, tuple[str, bool]] = {}

    def fake_tokenizer_from_pretrained(model_dir: str, local_files_only: bool = False):
        captured_calls["tokenizer"] = (model_dir, local_files_only)
        return object()

    def fake_model_from_pretrained(model_dir: str, local_files_only: bool = False):
        captured_calls["model"] = (model_dir, local_files_only)
        return _FakeLoadedT5Model()

    monkeypatch.setattr(t5_text_encoder_module.os.path, "isdir", lambda path: path == t5_text_encoder_module.DEFAULT_T5_MODEL_DIR)
    monkeypatch.setattr(t5_text_encoder_module.T5TokenizerFast, "from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr(t5_text_encoder_module.T5EncoderModel, "from_pretrained", fake_model_from_pretrained)

    encoder = t5_text_encoder_module.CosmosT5TextEncoder(device="cpu")

    assert captured_calls["tokenizer"] == (t5_text_encoder_module.DEFAULT_T5_MODEL_DIR, True)
    assert captured_calls["model"] == (t5_text_encoder_module.DEFAULT_T5_MODEL_DIR, True)
    assert encoder.device == "cpu"
    assert encoder.text_encoder.device == "cpu"
    assert encoder.text_encoder.eval_called is True


def test_base_pipeline_uses_same_fixed_t5_dir(monkeypatch) -> None:
    """主 pipeline 也必须和编码器默认值共用同一份常量."""

    t5_text_encoder_module, pipeline_module = _import_pipeline_module(monkeypatch)
    captured_init: dict[str, str] = {}

    class _FakeCosmosT5TextEncoder:
        def __init__(self, model_name: str, cache_dir: str):
            captured_init["model_name"] = model_name
            captured_init["cache_dir"] = cache_dir

    class _MinimalPipeline(pipeline_module.BaseWorldGenerationPipeline):
        """只保留 T5 加载路径,把其余重逻辑都 stub 掉."""

        def _load_model(self, checkpoint_name: str | None = None):
            self.model = None
            return None

        def _load_network(self):
            return None

        def _load_tokenizer(self):
            return None

        def _run_model(self, *args, **kwargs):
            return None

    monkeypatch.setattr(pipeline_module, "CosmosT5TextEncoder", _FakeCosmosT5TextEncoder)

    _MinimalPipeline(
        has_text_input=False,
        disable_guardrail=True,
        offload_network=True,
        offload_tokenizer=True,
        offload_text_encoder_model=False,
    )

    assert captured_init == {
        "model_name": t5_text_encoder_module.DEFAULT_T5_MODEL_NAME,
        "cache_dir": t5_text_encoder_module.DEFAULT_T5_MODEL_DIR,
    }
