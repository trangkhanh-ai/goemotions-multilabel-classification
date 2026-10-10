"""Kiểm xử lý AMP trên batch train thật; không tạo dữ liệu/metrics thực nghiệm giả."""
from src.models.distilbert_study import ROOT, read_json, prepare_frames, EmotionDataset, make_loader, load_bundle, write_json
from scripts.distilbert.train_distilbert import optimizer_update
import torch


def main():
    torch.set_num_threads(4)
    config=read_json(ROOT/'configs/distilbert_pilot.json')
    frames,_,_=prepare_frames(config)
    bundle=load_bundle(ROOT/'data/processed/c3_distilbert/pilot/seed_42','cuda')
    model=bundle['model'].train()
    dataset=EmotionDataset(frames['train'].head(4),bundle['tokenizer'],128)
    batch={k:v.cuda() for k,v in next(iter(make_loader(dataset,bundle['tokenizer'],4))).items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    scaler=torch.amp.GradScaler('cuda',init_scale=2.0**40)
    before=model.classifier.weight.detach().clone()
    schedule_before=scheduler.last_epoch
    with torch.autocast('cuda',dtype=torch.float16):
        loss=model(**batch).loss
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    updated=optimizer_update(model,optimizer,scheduler,scaler,1.0)
    assert not updated and scaler.get_scale()<2.0**40
    assert scheduler.last_epoch==schedule_before
    torch.testing.assert_close(before,model.classifier.weight,rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    with torch.autocast('cuda',dtype=torch.float16):
        loss=model(**batch).loss
    scaler.scale(loss).backward()
    assert optimizer_update(model,optimizer,scheduler,scaler,1.0)
    assert scheduler.last_epoch==schedule_before+1
    assert not torch.equal(before,model.classifier.weight)
    write_json(ROOT/'reports/c3_distilbert/amp_regression_check.json',{
        'status':'passed','real_train_ids':frames['train'].id.head(4).tolist(),
        'purpose':'Numerical error-handling check only; these modified weights are discarded',
        'overflow_check':'forced high loss scale on real batch; skipped weights and LR; scale reduced; gradients cleared',
        'finite_check':'normal scale on real batch updated weights and LR', 'test_used':False})
    print('PASS: AMP overflow skips weights/LR, then finite gradients update normally.')


if __name__=='__main__':
    main()
