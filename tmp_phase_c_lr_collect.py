from pathlib import Path
import json
root = Path("/workspace/lyra/outputs/refine_v2")
runs = sorted(root.glob("full_view_sr_stage3sr_phaseC_hr32_lr*_sub8_iter8_20260310"))
for run in runs:
    diag = run / "diagnostics.json"
    print(f"--- {run.name} ---")
    if not diag.exists():
        print("missing diagnostics")
        continue
    payload = json.loads(diag.read_text(encoding="utf-8"))
    final = payload.get("final", {})
    print("phase_reached", payload.get("phase_reached"))
    for key in ["psnr","residual_mean","sharpness","loss_total","loss_hr_rgb","loss_lr_consistency","psnr_hr","residual_mean_hr","psnr_native_render","residual_mean_native_render"]:
        print(key, final.get(key))
