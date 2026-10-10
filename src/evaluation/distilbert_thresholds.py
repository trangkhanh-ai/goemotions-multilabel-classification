"""Ngưỡng C3 thuộc đúng checkpoint; không thay đổi scores hoặc metadata huấn luyện."""
from pathlib import Path
import numpy as np
from src.models.distilbert_study import read_json, predict_texts, format_predictions
from src.datasets.goemotions import sha256
from src.evaluation.metrics import validate_thresholds


def load_c3_thresholds(run, metadata, mode='fixed'):
    if mode == 'fixed':
        return 0.5
    if mode != 'per_label':
        raise ValueError('threshold mode phải là fixed hoặc per_label')
    run = Path(run)
    saved = read_json(run/'thresholds_validation.json')
    expected = {'run_metadata_sha256':sha256(run/'run.json'), 'seed':metadata['config']['seed'],
                'label_names':metadata['label_names'], 'data_revision':metadata['data_revision'],
                'model_sha256':metadata['artifact_sha256']['best/model.safetensors'],
                'scores_sha256':metadata['artifact_sha256']['validation_scores.npz']}
    if any(saved.get(k)!=v for k,v in expected.items()):
        raise ValueError('Ngưỡng không thuộc đúng run/checkpoint/scores/mapping này')
    return validate_thresholds(saved['thresholds'], len(metadata['label_names']))


def predict_c3(bundle, texts, run, mode='fixed'):
    thresholds = load_c3_thresholds(run, bundle['metadata'], mode)
    results = predict_texts(bundle, texts)
    names = bundle['metadata']['label_names']
    scores = np.array([[item['scores'][label] for label in names] for item in results])
    return format_predictions(texts, scores, names, [item['original_tokens'] for item in results],
                              bundle['metadata']['config']['max_length'], thresholds)
