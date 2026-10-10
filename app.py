"""Demo D dùng đúng checkpoint C full được chọn từ ba kiến trúc × ba seed.

Chạy: python app.py --device auto
Không có checkpoint đầy đủ thì báo cách tạo; không trả kết quả giả.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from src.models.transformer import load_demo_selection, resolve_device, configure_console

ROOT = Path(__file__).resolve().parent


class TransformerDemo:
    def __init__(self, selection, device="auto", thresholds=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        folder, self.metadata, self.thresholds = load_demo_selection(
            ROOT, selection, thresholds_path=thresholds
        )
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(folder / "checkpoint", local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            folder / "checkpoint", local_files_only=True, trust_remote_code=False
        ).to(self.device).eval()
        self.labels = self.metadata["label_names"]
        self.max_length = self.metadata["config"]["max_length"]
        if self.model.config.num_labels != 28:
            raise ValueError("Checkpoint demo không có head 28 nhãn")
        self.torch = torch

    def predict(self, text):
        """Cùng tokenizer/mapping/sigmoid/ngưỡng như lúc đánh giá checkpoint."""
        text = (text or "").strip()
        if not text:
            return "Vui lòng nhập một câu hoặc bình luận tiếng Anh.", []
        if len(text) > 20000:
            return "Văn bản quá dài. Vui lòng dùng một bình luận dưới 20.000 ký tự.", []
        # Chỉ đếm để báo cắt token; chuỗi dài này không được đưa vào encoder.
        raw_tokens = self.tokenizer(text, truncation=False, add_special_tokens=True,
                                    verbose=False)["input_ids"]
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            context = self.torch.autocast("cuda", dtype=self.torch.float16) if self.device.type == "cuda" else nullcontext()
            with context:
                logits = self.model(**inputs).logits
            scores = self.torch.sigmoid(logits.float())[0].cpu().numpy()
        if scores.shape != (28,) or not np.isfinite(scores).all():
            raise ValueError("Mô hình trả scores sai")
        limits = np.broadcast_to(np.asarray(self.thresholds), (28,))
        selected = [name for name, score, limit in zip(self.labels, scores, limits) if score >= limit]
        message = ("**Cảm xúc dự đoán:** " + ", ".join(selected)) if selected else (
            "**Chưa có nhãn nào đạt ngưỡng.** Không tự suy ra neutral trong trường hợp này."
        )
        if len(raw_tokens) > self.max_length:
            message += f"\n\nVăn bản đã được cắt còn {self.max_length} token để dự đoán."
        rows = [[self.labels[i], round(float(scores[i]), 4), round(float(limits[i]), 4),
                 "Có" if scores[i] >= limits[i] else ""] for i in np.argsort(-scores)]
        return message, rows


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "data/processed/transformers/selected_model.json")
    parser.add_argument("--thresholds", type=Path, help="Ngưỡng validation của đúng run C được chọn")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.selection.is_file():
        parser.error("Chưa có best C. Chạy đủ ba kiến trúc × ba seed, rồi "
                     "python -m scripts.transformers.select_best_transformer. Xem docs/EXPERIMENTS.md.")
    import gradio as gr
    predictor = TransformerDemo(args.selection, args.device, args.thresholds)
    with gr.Blocks(title="GoEmotions — nhận diện cảm xúc") as demo:
        gr.Markdown("# Nhận diện cảm xúc trong bình luận\n"
                    "Nhập văn bản **tiếng Anh**. Một câu có thể có nhiều cảm xúc trong 28 nhãn GoEmotions.")
        gr.Markdown(f"Mô hình: **{predictor.metadata['architecture']}**. "
                    "Điểm là đầu ra của mô hình, chưa được hiệu chuẩn thành xác suất đáng tin cậy.")
        text = gr.Textbox(label="Bình luận tiếng Anh", lines=4,
                          placeholder="Thank you so much! I am really happy with your help.")
        button = gr.Button("Nhận diện cảm xúc", variant="primary")
        result = gr.Markdown()
        table = gr.Dataframe(headers=["Cảm xúc", "Điểm", "Ngưỡng", "Được chọn"],
                             datatype=["str", "number", "number", "str"], interactive=False,
                             label="Điểm của 28 nhãn")
        button.click(predictor.predict, inputs=text, outputs=[result, table], concurrency_limit=1)
        gr.Examples(examples=[["Thank you so much! I am really happy with your help."],
                              ["I am disappointed and worried about what will happen next."],
                              ["The meeting starts at nine tomorrow."]], inputs=text)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
