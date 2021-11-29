# optimizer
optimizer = dict(
    type="Adam",
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[85, 100],
)
total_epochs = 105
