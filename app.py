import os
import gradio as gr
from models.model import Model
from utils.string import normalize_model_name

def get_model_dirs():
    base = "models"
    return [
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
        and {"model.pth", "itos.pkl", "stoi.pkl"}.issubset(set(os.listdir(os.path.join(base, d))))
    ]

model_dirs = get_model_dirs()
display_to_dir = {
    normalize_model_name(d): d
    for d in model_dirs
}
display_choices = list(display_to_dir.keys())

def predict_sentiment(review: str, model_dir: str):
    mdl = Model(display_to_dir[model_dir])
    mdl.load()
    return mdl.predict(review)

with gr.Blocks() as demo:
    gr.Markdown("## 📝 Product Review Sentiment Classifier")

    model_dropdown = gr.Dropdown(
        choices=display_choices,
        label="Select Model",
        value=display_choices[0] if display_choices else None
    )

    review_input = gr.Textbox(
        lines=3,
        placeholder="Enter your product review here…",
        label="Review"
    )
    output_label    = gr.Label(label="Prediction")
    classify_button = gr.Button("Classify")

    classify_button.click(
        fn=predict_sentiment,
        inputs=[review_input, model_dropdown],
        outputs=output_label
    )

demo.launch()
