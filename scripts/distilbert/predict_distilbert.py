"""Suy luận bằng checkpoint đã fine-tune; dùng chung hàm với demo."""
import argparse
import json
from src.models.distilbert_study import ROOT, read_json, load_bundle
from src.evaluation.distilbert_thresholds import predict_c3


def main():
    parser = argparse.ArgumentParser()
    selected=ROOT/'reports/c3_distilbert/full/representative_c3.json'
    default_run=ROOT/read_json(selected)['run'] if selected.exists() else ROOT/'data/processed/c3_distilbert/pilot/seed_42'
    parser.add_argument("--run", default=str(default_run))
    parser.add_argument('--threshold-mode',choices=['fixed','per_label'],default='fixed')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    bundle = load_bundle(args.run, args.device)
    print(json.dumps({"mode": bundle["metadata"]["mode"],
                      'threshold_mode':args.threshold_mode,
                      "prediction": predict_c3(bundle, [args.text], args.run, args.threshold_mode)[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
