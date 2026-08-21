from vllm import ModelRegistry

DSPARK_MODEL_ARCHITECTURE = "DSparkDraftModel"
DSPARK_MODEL_CLASS = "vllm_ascend.models.deepseek_v4_dspark:DSparkDeepseekV4ForCausalLM"


def register_dspark_model() -> None:
    """Register the Ascend DSpark model without re-importing model code."""
    registered_model = ModelRegistry.models.get(DSPARK_MODEL_ARCHITECTURE)
    if (
        getattr(registered_model, "module_name", None) == "vllm_ascend.models.deepseek_v4_dspark"
        and getattr(registered_model, "class_name", None) == "DSparkDeepseekV4ForCausalLM"
    ):
        return
    ModelRegistry.register_model(DSPARK_MODEL_ARCHITECTURE, DSPARK_MODEL_CLASS)


def register_model():
    ModelRegistry.register_model("DeepseekV4ForCausalLM", "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")

    ModelRegistry.register_model("DeepSeekV4MTPModel", "vllm_ascend.models.deepseek_v4_mtp:DeepSeekV4MTP")
    register_dspark_model()
    ModelRegistry.register_model(
        "LlamaForCausalLMVwnEagle3", "vllm_ascend.models.llama_eagle3_vwn:Eagle3VwnLlamaForCausalLM"
    )
