"""Demo cục bộ của Huy. Chạy: python -m streamlit run app.py"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import streamlit as st
from src.models.distilbert_study import ROOT, load_bundle, predict_texts
from src.models.distilbert_study import read_json
from src.evaluation.distilbert_thresholds import predict_c3, load_c3_thresholds


def run_demo_prediction(bundle, text, run=None, mode='fixed'):
    """Cùng hàm với CLI; không tự gán neutral hoặc ép chọn một nhãn."""
    return predict_c3(bundle, [text], run, mode)[0]


@st.cache_resource
def cached_bundle(path, stamp, device):
    return load_bundle(path, device)


def main():
    st.set_page_config(page_title="GoEmotions — Nhật Huy", page_icon="💬", layout="centered")
    st.title("Cảm xúc trong câu chữ")
    st.write("Nhập một bình luận tiếng Anh để xem những cảm xúc mô hình nhận diện.")
    st.caption("Một câu có thể mang nhiều nhãn; bảng bên dưới hiển thị điểm của đủ 28 nhãn.")
    selected = ROOT/'reports/c3_distilbert/full/representative_c3.json'
    default_run = ROOT/read_json(selected)['run'] if selected.exists() else ROOT/'data/processed/c3_distilbert/pilot/seed_42'
    default = os.environ.get("C3_DEMO_RUN", str(default_run))
    path = st.sidebar.text_input("Thư mục run đã huấn luyện", value=default)
    st.sidebar.caption("Demo hiện hỗ trợ bundle C3. Tích hợp checkpoint BERT/RoBERTa nếu được chọn sẽ thực hiện khi nhận bàn giao.")
    mode_label = st.sidebar.selectbox('Ngưỡng dự đoán', ['Cơ sở: 0,5', 'Từng nhãn: chọn trên validation'])
    mode = 'fixed' if mode_label == 'Cơ sở: 0,5' else 'per_label'
    device = os.environ.get("C3_DEMO_DEVICE", "cpu")
    examples_path=ROOT/'reports/c3_distilbert/full/demo_examples.json'
    example=read_json(examples_path)[0] if examples_path.exists() else None
    text = st.text_area("Bình luận tiếng Anh", value=example['text'] if example else '', max_chars=20000, height=140)
    if example:
        st.caption(f"Câu mặc định là mẫu validation thật, ID {example['id']}; bạn có thể nhập câu khác.")
    if not st.button("Phân tích cảm xúc", type="primary"):
        return
    if not text.strip():
        st.warning("Hãy nhập một bình luận trước khi phân tích.")
        return
    try:
        run_path = Path(path)
        with st.spinner("Đang đọc bình luận…"):
            bundle = cached_bundle(str(run_path.resolve()), (run_path/"run.json").stat().st_mtime_ns, device)
            result = run_demo_prediction(bundle, text, run_path, mode)
        meta = bundle["metadata"]
        if meta["mode"] == "pilot":
            st.warning("Bản chạy thử kỹ thuật: mô hình mới học trên một tập nhỏ, chưa dùng để đánh giá chất lượng hoặc làm demo cuối kỳ.")
        else:
            st.info('Demo phần C3 của Huy; mô hình thắng giữa C1/C2/C3 chưa được xác định.')
        st.caption(f"{meta['method']} · seed {meta['config']['seed']} · {mode_label} · chế độ {meta['mode']}")
        st.caption(f"Checkpoint: {meta['config']['model_id']} · revision {meta['config']['revision'][:12]}")
        if mode == 'per_label':
            st.caption('Ngưỡng được chọn riêng từ validation của checkpoint này; kết quả tuned-validation không phải đánh giá độc lập.')
        if result["truncated"]:
            st.info(f"Câu có {result['original_tokens']} token; mô hình chỉ xử lý tối đa {result['max_length']} token.")
        if result["labels"]:
            st.subheader("Nhãn đạt ngưỡng")
            st.write(", ".join(result["labels"]))
        else:
            st.info("Chưa có nhãn nào đạt ngưỡng. Không tự gán neutral trong trường hợp này.")
        limits=np.broadcast_to(load_c3_thresholds(run_path,meta,mode),(len(meta['label_names']),))
        table = pd.DataFrame([{"Nhãn":k,"Điểm":v,"Ngưỡng":float(t),"Đạt ngưỡng":k in result["labels"]}
                              for (k,v),t in zip(result["scores"].items(),limits)])
        st.dataframe(table.sort_values("Điểm", ascending=False), hide_index=True,
                     column_config={"Điểm":st.column_config.NumberColumn(format="%.4f"),
                                    "Ngưỡng":st.column_config.NumberColumn(format="%.2f")})
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        st.error(f"Chưa thể dự đoán: {exc}")


if __name__ == "__main__":
    main()
