# optimizer
optimizer = dict(type='Adam', lr=0.003, weight_decay=0.0001)
optimizer_config = dict()
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr=3e-5,
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU')