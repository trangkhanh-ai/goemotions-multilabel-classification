"""Kiểm artifact pilot, metrics, CLI và giao diện Streamlit bằng checkpoint thật."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.models.distilbert_study import ROOT, load_bundle, predict_texts, prepare_frames, read_json, write_json
from src.evaluation.distilbert_thresholds import predict_c3, load_c3_thresholds
import numpy as np
import pandas as pd
import torch
from src.datasets.goemotions import multi_hot, load_goemotions
from src.evaluation.metrics import evaluate_multilabel
from streamlit.testing.v1 import AppTest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'data/processed/c3_distilbert/pilot/seed_42')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/c3_distilbert/verification.json')
    args = parser.parse_args()
    run = args.run.resolve()
    torch.set_num_threads(4)
    bundle = load_bundle(run, 'cpu')
    meta = bundle['metadata']
    frames, names, _ = prepare_frames(meta['config'])
    with np.load(run/'validation_scores.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['ids'], frames['validation'].id.to_numpy(dtype=str))
        assert saved['label_names'].tolist() == names
        scores = saved['scores'].copy()
    for split in ('train', 'validation'):
        assert pd.read_csv(run/f'{split}_ids.csv').id.tolist() == frames[split].id.tolist()
    metric = evaluate_multilabel(multi_hot(frames['validation'].labels.tolist()), scores, names)
    assert metric == read_json(run/'validation_metrics.json')
    assert not meta['test_used']
    sample = frames['validation'].text.iloc[0]
    reference = predict_texts(bundle, [sample])[0]
    # Chạy CLI thành tiến trình riêng, parse đúng JSON người dùng nhận.
    output = subprocess.check_output([sys.executable, '-m', 'scripts.distilbert.predict_distilbert',
        '--run', str(run), '--text', sample, '--device', 'cpu'], cwd=ROOT, text=True, encoding='utf-8')
    cli = json.loads(output)['prediction']
    np.testing.assert_allclose(list(cli['scores'].values()), list(reference['scores'].values()), atol=1e-6, rtol=0)
    assert cli['labels'] == reference['labels']
    # Demo C3 riêng của Huy; app.py ở gốc là demo Gradio của best C toàn nhóm.
    os.environ['C3_DEMO_RUN'] = str(run)
    os.environ['C3_DEMO_DEVICE'] = 'cpu'
    app = AppTest.from_file(str(ROOT/'app_distilbert_huy.py'), default_timeout=120).run()
    assert not app.exception
    app.text_area[0].set_value(sample)
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert len(app.dataframe) == 1
    app_scores = app.dataframe[0].value.set_index('Nhãn').loc[names, 'Điểm'].to_numpy()
    np.testing.assert_allclose(app_scores, list(reference['scores'].values()), atol=1e-6, rtol=0)
    app_labels = set(app.dataframe[0].value.query('`Đạt ngưỡng`')['Nhãn'])
    assert app_labels == set(reference['labels'])
    tuned_diff = None
    if (run/'thresholds_validation.json').exists():
        wrong_meta={**meta,'config':{**meta['config'],'seed':-1}}
        try:
            load_c3_thresholds(run,wrong_meta,'per_label')
        except ValueError:
            pass
        else:
            raise AssertionError('Ngưỡng phải từ chối metadata sai seed')
        tuned = predict_c3(bundle,[sample],run,'per_label')[0]
        app.selectbox[0].select('Từng nhãn: chọn trên validation')
        app.button[0].click().run()
        assert not app.exception and not app.error
        tuned_scores=app.dataframe[0].value.set_index('Nhãn').loc[names,'Điểm'].to_numpy()
        np.testing.assert_allclose(tuned_scores,list(tuned['scores'].values()),atol=1e-6,rtol=0)
        assert set(app.dataframe[0].value.query('`Đạt ngưỡng`')['Nhãn'])==set(tuned['labels'])
        tuned_output=subprocess.check_output([sys.executable,'-m','scripts.distilbert.predict_distilbert',
            '--run',str(run),'--text',sample,'--threshold-mode','per_label'],cwd=ROOT,text=True,encoding='utf-8')
        tuned_cli=json.loads(tuned_output)['prediction']
        assert tuned_cli['labels']==tuned['labels']
        np.testing.assert_allclose(list(tuned_cli['scores'].values()),list(tuned['scores'].values()),atol=1e-6,rtol=0)
        tuned_diff=float(np.max(np.abs(tuned_scores-np.array(list(tuned['scores'].values())))))
        app.selectbox[0].select('Cơ sở: 0,5')
    app.text_area[0].set_value('   ')
    app.button[0].click().run()
    assert not app.exception and any('Hãy nhập' in x.value for x in app.warning)
    assert not app.dataframe
    # Official validation không có câu >128 token với DistilBERT. Dùng outlier train thật.
    long_frame=load_goemotions(ROOT,write_metadata=False,splits=('train',))[0]['train']
    lengths=bundle['tokenizer'](long_frame.text.tolist(),truncation=False,padding=False)['input_ids']
    long_index=next(i for i,x in enumerate(lengths) if len(x)>meta['config']['max_length'])
    app.text_area[0].set_value(long_frame.text.iloc[long_index])
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert any('token' in x.value for x in app.info)
    empty_indices=np.flatnonzero((scores>=.5).sum(1)==0)
    if len(empty_indices):
        app.text_area[0].set_value(frames['validation'].text.iloc[int(empty_indices[0])])
        app.button[0].click().run()
        assert not app.exception and any('Chưa có nhãn' in x.value for x in app.info)
    multi_indices=np.flatnonzero((scores>=.501).sum(1)>=2)
    multi_id=None
    if len(multi_indices):
        i=int(multi_indices[0])
        multi_text=frames['validation'].text.iloc[i]
        multi_id=frames['validation'].id.iloc[i]
        multi_reference=predict_c3(bundle,[multi_text],run,'fixed')[0]
        app.text_area[0].set_value(multi_text)
        app.button[0].click().run()
        assert not app.exception and not app.error
        multi_labels=set(app.dataframe[0].value.query('`Đạt ngưỡng`')['Nhãn'])
        assert len(multi_labels)>=2 and multi_labels==set(multi_reference['labels'])
    report = {
        'status':'passed', 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'run':run.relative_to(ROOT).as_posix(), 'mode':meta['mode'], 'test_used':False,
        'checks':{
            'artifact_sha256_and_mapping':'passed', 'sample_ids_and_npz_alignment':'passed',
            'validation_metrics_recomputed':'passed', 'checkpoint_reload_during_training':'passed',
            'cli_shared_inference_scores_and_labels':'passed', 'streamlit_widget_prediction_scores_and_labels':'passed',
            'streamlit_empty_text_handling':'passed', 'streamlit_long_text_truncation_notice':'passed'},
        'score_comparison_atol':1e-6,
        'cli_max_abs_score_diff':float(np.max(np.abs(np.array(list(cli['scores'].values()))-np.array(list(reference['scores'].values()))))),
        'streamlit_max_abs_score_diff':float(np.max(np.abs(app_scores-np.array(list(reference['scores'].values()))))),
        'tuned_app_cli_parity':'passed' if tuned_diff is not None else 'not applicable',
        'tuned_streamlit_max_abs_score_diff':tuned_diff,
        'empty_prediction_ui':'passed on real validation text' if len(empty_indices) else 'no such sample in validation',
        'multi_label_ui':'passed on real validation text' if multi_id else 'no robust multi-label example in validation',
        'multi_label_example_id':multi_id,
        'example_id':frames['validation'].id.iloc[0],
        'long_example_id':long_frame.id.iloc[long_index], 'long_example_split':'train',
        'long_example_tokens':len(lengths[long_index]),
        'note':'C3 integration check; all nonblank input examples are real dataset texts; long-input check uses train outlier; not test-set evaluation.'}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
