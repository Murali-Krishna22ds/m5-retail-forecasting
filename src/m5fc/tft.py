"""Temporal Fusion Transformer (PyTorch Forecasting) on the bottom series with native quantile loss.
Encoder 56 d, decoder 28 d. Known-future inputs: price, promo, SNAP, calendar. Static: item/store/dept/cat/state."""
from __future__ import annotations
import numpy as np, pandas as pd, torch
from .config import LAST_TRAIN_D, CAL_START, CAL_END, TEST_START, TEST_END, QUANTILES, RES, SEED, H
from .store import save, qcols

ENC, T_FROM = 56, 1500


def prepare(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt[bt.d >= T_FROM].copy()
    d["time_idx"] = d.d.astype(int)
    d["dow"] = d.wday.astype(int).astype(str); d["event_type"] = d.event_type.astype(int).astype(str)
    for c in ["item_id", "store_id", "dept_id", "cat_id", "state_id"]: d[c] = d[c].astype(str)
    d["price_rel_item"] = d.price_rel_item.fillna(0); d["price_ratio"] = d.price_ratio.fillna(0)
    d["promo"] = d.promo.fillna(0); d["avail"] = d.avail.fillna(0)
    d["y"] = d.y.astype("float32")
    keep = ["sku", "time_idx", "d", "y", "item_id", "store_id", "dept_id", "cat_id", "state_id", "dow", "event_type",
            "price_ratio", "price_rel_item", "promo", "avail", "snap", "is_event", "days_to_event", "days_since_event", "dom"]
    return d[keep].reset_index(drop=True)


def run(bt: pd.DataFrame, epochs: int = 6, batches_per_epoch: int = 300, batch_size: int = 128, log=print):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    torch.set_num_threads(1); pl.seed_everything(SEED)
    data = prepare(bt)
    mk = dict(time_idx="time_idx", target="y", group_ids=["sku"], min_encoder_length=28, max_encoder_length=ENC,
              min_prediction_length=H, max_prediction_length=H,
              static_categoricals=["item_id", "store_id", "dept_id", "cat_id", "state_id"],
              time_varying_known_categoricals=["dow", "event_type"],
              time_varying_known_reals=["time_idx", "price_ratio", "price_rel_item", "promo", "avail", "snap",
                                        "is_event", "days_to_event", "days_since_event", "dom"],
              time_varying_unknown_reals=["y"],
              target_normalizer=GroupNormalizer(groups=["sku"], transformation="softplus"),
              add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True)
    train = TimeSeriesDataSet(data[data.time_idx <= LAST_TRAIN_D], **mk)
    val = TimeSeriesDataSet.from_dataset(train, data[data.time_idx <= CAL_END], predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(train, data[data.time_idx <= TEST_END], predict=True, stop_randomization=True)
    tl = train.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    vl = val.to_dataloader(train=False, batch_size=256, num_workers=0)
    log(f"train windows={len(train):,}  val series={len(val)}  test series={len(test)}")
    model = TemporalFusionTransformer.from_dataset(
        train, learning_rate=0.01, hidden_size=16, attention_head_size=2, dropout=0.1, hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=QUANTILES), optimizer="Adam", reduce_on_plateau_patience=2, log_interval=-1)
    log(f"TFT parameters: {sum(p.numel() for p in model.parameters()):,}")
    trainer = pl.Trainer(max_epochs=epochs, accelerator="cpu", gradient_clip_val=0.1, limit_train_batches=batches_per_epoch,
                         enable_progress_bar=False, enable_model_summary=False, logger=False, enable_checkpointing=True,
                         default_root_dir=str(RES / "tft_ckpt"),
                         callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")])
    trainer.fit(model, tl, vl)
    best = TemporalFusionTransformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    hist = {k: float(v) for k, v in trainer.callback_metrics.items()}
    log("final metrics", hist)
    out = []
    for win, ds in (("cal", val), ("test", test)):
        dl = ds.to_dataloader(train=False, batch_size=256, num_workers=0)
        pr = best.predict(dl, mode="quantiles", return_index=True)
        arr = pr.output.numpy() if hasattr(pr, "output") else pr[0].numpy()          # n x H x Q
        idx = pr.index if hasattr(pr, "index") else pr[1]
        start = idx["time_idx"].values; sku = idx["sku"].values
        for j in range(H):
            o = pd.DataFrame({"d": start + j, "node": sku, "window": win})
            q = np.sort(np.clip(arr[:, j, :], 0, None), axis=1)
            for i, qq in enumerate(QUANTILES): o[f"q{qq}"] = q[:, i]
            o["point"] = o["q0.5"]
            out.append(o)
    fc = pd.concat(out, ignore_index=True)
    fc = fc[fc.d.between(CAL_START, TEST_END)]
    # off-shelf rows -> 0
    av = bt[["sku", "d", "avail"]].rename(columns={"sku": "node"})
    fc = fc.merge(av, on=["node", "d"], how="left")
    for c in ["point"] + qcols(): fc.loc[fc.avail != 1, c] = 0.0
    fc = fc.drop(columns="avail")
    save("tft", fc)
    return fc, hist
