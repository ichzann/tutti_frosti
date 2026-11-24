"""FTW 3-class boundary probability on small 8-band windows, run in the separate `ftw` conda env.

    <ftw env python> validation/brazil_psr/ftw_infer.py <in.tif> [<in.tif> ...]

Writes <stem minus _in>_prob.tif next to each input: float32 softmax (background, interior, boundary),
mean of the CCBY and FULL 3-class checkpoints.  Preprocessing as ftw.datamodules.preprocess (/3000), native
10 m (no upsampling: on 2 km windows the 2x / 3x outputs flip between near-empty and near-full interior,
while the 1x boundary class traces the visible edges), whole window as one zero-padded patch.
"""
import sys
from pathlib import Path

import rasterio
import torch
import torch.nn.functional as F
from ftw.trainers import CustomSemanticSegmentationTask

D = Path(__file__).resolve().parents[2] / "cache" / "brazil_psr" / "ftw_models"
models = []
for c in ["CCBY", "FULL"]:
    t = CustomSemanticSegmentationTask.load_from_checkpoint(D / f"3_Class_{c}_FTW_Pretrained.ckpt", map_location="cpu")
    t.freeze()
    models.append(t.model.eval())
for fp in sys.argv[1:]:
    with rasterio.open(fp) as src:
        x = src.read().astype("float32") / 3000.0
        prof = src.profile
    _, H, W = x.shape
    t = torch.from_numpy(x)[None]
    t = F.pad(t, (0, (-W) % 32, 0, (-H) % 32), mode="reflect")
    with torch.inference_mode():
        p = sum(torch.softmax(m(t), 1) for m in models)[0, :, :H, :W].numpy() / len(models)
    prof.update(count=3, dtype="float32")
    with rasterio.open(fp.replace("_in.tif", "_prob.tif"), "w", **prof) as dst:
        dst.write(p)
    print(Path(fp).name, "boundary share", round(float((p.argmax(0) == 2).mean()), 3), flush=True)
