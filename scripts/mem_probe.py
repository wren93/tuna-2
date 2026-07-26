"""Measure GPU memory cost (and step time) of Tuna t2i inference, no checkpoint load.

Covers all three variants; per-variant sampling defaults mirror
``scripts/launch/predict.sh`` so the numbers match a real predict run.

    python scripts/mem_probe.py                     # tuna-2 (default)
    python scripts/mem_probe.py tuna-r
    python scripts/mem_probe.py tuna --size 2b
    python scripts/mem_probe.py tuna-2 inference.height=1024 inference.width=1024

Any extra argument is passed through to hydra verbatim, so anything in
``configs/predict/t2i.yaml`` can be overridden on the command line.
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # so `tuna` imports without PYTHONPATH

from tuna.inference.runner import TunaInference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mem_probe")

CONFIG_DIR = os.environ.get("TUNA_PREDICT_CONFIG_DIR", str(REPO_ROOT / "configs" / "predict"))
PROMPT = "a photo of a cat sitting on a windowsill"
N_PASSES = 2  # first pass is cold (warmup), second is steady state

# Per-variant defaults, kept in sync with the t2i branch of predict.sh.
# `models` maps --size to the model config name.
VARIANTS = {
    # Variant C — pure patchify, no encoder ("none_encoder" in predict.sh).
    "tuna-2": {
        "models": {"7b": "tuna_2_pixel_7b"},
        "overrides": [
            "inference.pipe=Tuna2PixelPipeline",
            "inference.generation_mode=t2i_pixel",
            "inference.guidance_scale=3",
            "inference.sampling_method=euler",
            "inference.noise_scale=8",
        ],
        "needs": "Qwen2.5-7B-Instruct",
    },
    # Variant B — SigLIP-only pixel diffusion ("siglip_pixel").
    "tuna-r": {
        "models": {"7b": "tuna_2r_pixel_7b"},
        "overrides": [
            "inference.pipe=Tuna2RPixelPipeline",
            "inference.generation_mode=t2i_pixel",
            "inference.guidance_scale=4",
            "inference.sampling_method=heun",
        ],
        "needs": "Qwen2.5-7B-Instruct + siglip2-so400m-patch16-512 (config.json is enough; "
                 "weights only warm-init the encoder and failure is non-fatal)",
    },
    # Variant A — WAN 2.2 VAE latent diffusion ("vae").
    "tuna": {
        "models": {"7b": "tuna_7b", "2b": "tuna_2b"},
        "overrides": [
            "inference.pipe=TunaPipeline",
            "inference.generation_mode=t2i",
            "inference.guidance_scale=7.5",
        ],
        "needs": "Qwen2.5-{7B,1.5B}-Instruct + siglip2-so400m-patch16-512 + "
                 "Wan-AI/Wan2.2-T2V-5B (the VAE is a hard requirement — it is built, "
                 "not warm-init, so a missing cache entry aborts the run)",
    },
}
ALIASES = {  # predict.sh spellings
    "none_encoder": "tuna-2",
    "siglip_pixel": "tuna-r",
    "vae": "tuna",
    "tuna2": "tuna-2",
    "tuna_r": "tuna-r",
}


def mib(x: int) -> str:
    return f"{x / 1024 ** 2:,.0f} MiB"


def report(tag: str) -> None:
    log.info(
        "[MEM] %-22s alloc=%s  peak_alloc=%s  reserved=%s  peak_reserved=%s",
        tag,
        mib(torch.cuda.memory_allocated()),
        mib(torch.cuda.max_memory_allocated()),
        mib(torch.cuda.memory_reserved()),
        mib(torch.cuda.max_memory_reserved()),
    )


def smi() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out or "(no compute apps reported)"


def parse_args() -> tuple[str, str, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "variant",
        nargs="?",
        default="tuna-2",
        help="tuna-2 (default) | tuna-r | tuna. predict.sh names "
             "(none_encoder / siglip_pixel / vae) also work.",
    )
    parser.add_argument(
        "--size", default="7b", help="Model size: 7b (default), or 2b for the `tuna` variant."
    )
    args, extra = parser.parse_known_args()

    variant = ALIASES.get(args.variant, args.variant)
    if variant not in VARIANTS:
        parser.error(
            f"unknown variant {args.variant!r}; choose from "
            f"{', '.join(VARIANTS)} (or {', '.join(ALIASES)})"
        )
    models = VARIANTS[variant]["models"]
    if args.size not in models:
        parser.error(
            f"variant {variant!r} has no --size {args.size!r}; available: {', '.join(models)}"
        )
    return variant, args.size, extra


def main() -> None:
    variant, size, extra = parse_args()
    spec = VARIANTS[variant]
    model_name = spec["models"][size]

    overrides = [
        f"model={model_name}",
        "inference.use_ckpt=false",
        "inference.height=512",
        "inference.width=512",
        "inference.num_inference_steps=50",
        "inference.output_dir=/tmp/unused",
    ]
    overrides += spec["overrides"]
    overrides += extra  # user overrides win — hydra keeps the last occurrence

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="t2i", overrides=overrides)

    log.info("Variant: %s (%s) | needs in HF cache: %s", variant, model_name, spec["needs"])
    log.info("Overrides: %s", overrides)
    torch.cuda.init()
    report("cuda-context-only")

    t0 = time.time()
    model = instantiate(cfg.model)
    n_params = sum(p.numel() for p in model.parameters())
    build_s = time.time() - t0
    log.info("Model built on CPU in %.1fs | params=%.3fB", build_s, n_params / 1e9)
    report("after-cpu-build")

    kwargs = OmegaConf.to_container(cfg.inference, resolve=True)
    kwargs.pop("output_dir", None)
    kwargs.pop("seed", None)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    runner = TunaInference(model=model, **kwargs)
    load_s = time.time() - t0
    log.info("Runner ready in %.1fs (weights on GPU, pipeline built)", load_s)
    report("weights-resident")
    weights_only = torch.cuda.memory_allocated()

    # Two passes: the first includes kernel autotuning / warmup, the second is
    # the steady-state number to quote.
    steps = int(kwargs["num_inference_steps"])
    torch.cuda.reset_peak_memory_stats()
    gen_times = []
    for i in range(N_PASSES):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            runner({"text": [PROMPT]}, seed=42)
        torch.cuda.synchronize()
        dt = time.time() - t0
        gen_times.append(dt)
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.current.sm,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout.strip()
        log.info("Generation pass %d/%d: %.1fs (%.0f ms/step) | gpu now: %s",
                 i + 1, N_PASSES, dt, dt * 1000 / steps, util)
    report("after-generation")

    peak = torch.cuda.max_memory_allocated()
    log.info("=" * 78)
    log.info("SUMMARY  %s (%s) | %s | %sx%s | %s | steps=%s | cfg=%s",
             variant, model_name, kwargs["pipe"],
             kwargs["height"], kwargs["width"], kwargs["weight_dtype"],
             kwargs["num_inference_steps"], kwargs["guidance_scale"])
    log.info("  params                : %.3fB", n_params / 1e9)
    log.info("  weights resident      : %s", mib(weights_only))
    log.info("  peak allocated (gen)  : %s", mib(peak))
    log.info("  activation headroom   : %s", mib(peak - weights_only))
    log.info("  peak reserved (torch) : %s", mib(torch.cuda.max_memory_reserved()))
    log.info("  cpu build             : %.1fs", build_s)
    log.info("  weights -> gpu + pipe : %.1fs", load_s)
    log.info("  gen cold / warm       : %s",
             " / ".join(f"{t:.1f}s ({t * 1000 / steps:.0f} ms/step)" for t in gen_times))
    log.info("  nvidia-smi process    : %s", smi())
    log.info("=" * 78)


if __name__ == "__main__":
    main()
