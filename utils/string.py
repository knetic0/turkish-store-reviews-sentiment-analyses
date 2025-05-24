def normalize_model_name(model_name: str) -> str:
    return model_name.replace(".pth", " ").replace("_", " ").title()