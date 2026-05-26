"""open-cam web UI — FastAPI + HTMX."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncGenerator

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from camera_model import load_camera_model  # noqa: E402

REPO = Path(__file__).parent.parent.resolve()
RECIPES_DIR = REPO / "config" / "camera_recipes"
OUT_DIR = REPO / "out"
ILLUMINANTS_DIR = REPO / "spectra" / "illuminant" / "interpolated"
LENSES_DIR = REPO / "config" / "lenses"

app = FastAPI(title="open-cam")
app.mount("/static/out", StaticFiles(directory=str(OUT_DIR)), name="out")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# in-process job store: job_id -> {"proc": asyncio.subprocess, "log": [str], "done": bool}
_jobs: dict[str, dict] = {}

BRAND_ORDER = [
    "Canon", "Nikon", "Sony", "Fujifilm", "Panasonic", "Olympus",
    "Leica", "Konica Minolta", "Apple", "Samsung", "Google", "Huawei", "Xiaomi", "Sigma",
    "Research", "Default",
]

BRAND_PREFIXES: dict[str, str] = {
    "canon": "Canon",
    "nikon": "Nikon",
    "sony": "Sony",
    "fujifilm": "Fujifilm",
    "panasonic": "Panasonic",
    "olympus": "Olympus",
    "leica": "Leica",
    "konica_minolta": "Konica Minolta",
    "iphone": "Apple",
    "samsung": "Samsung",
    "google": "Google",
    "huawei": "Huawei",
    "xiaomi": "Xiaomi",
    "sigma": "Sigma",
    "research": "Research",
    "study": "Research",
    "default": "Default",
}

# Get brand based on a cameras slug
def _brand(name: str) -> str:
    for prefix, brand in BRAND_PREFIXES.items():
        if name.startswith(prefix):
            return brand
    return "Other"


def _display(name: str) -> str:
    """Humanize a recipe slug: nikon_z6 → Nikon Z6."""
    return name.replace("_", " ").title()


def _grouped_cameras(query: str = "") -> list[dict]:
    names = sorted(p.stem for p in RECIPES_DIR.glob("*.yaml") if p.stem != "INDEX")
    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower() or q in _display(n).lower()]
    groups: dict[str, list] = defaultdict(list)
    for name in names:
        groups[_brand(name)].append({"name": name, "display": _display(name)})
    result = []
    for brand in BRAND_ORDER:
        if brand in groups:
            result.append({"brand": brand, "cameras": groups[brand]})
    for brand, cameras in groups.items():
        if brand not in BRAND_ORDER:
            result.append({"brand": brand, "cameras": cameras})
    return result

# Lensfile loading helper
def _lensfiles() -> list[dict]:
    if not LENSES_DIR.is_dir():
        return []
    files = sorted(LENSES_DIR.glob("*.dat"))
    result = []
    for f in files:
        stem = f.stem  # e.g. "dgauss.50mm"
        label = stem.replace(".", " ").replace("_", " ").title()
        result.append({"path": f"config/lenses/{f.name}", "label": label, "stem": stem})
    return result


def _illuminants() -> list[str]:
    if not ILLUMINANTS_DIR.is_dir():
        raise RuntimeError(f"Illuminants directory not found: {ILLUMINANTS_DIR}")
    return sorted(p.stem for p in ILLUMINANTS_DIR.glob("*.csv"))


def _illuminant_path(name: str) -> str:
    return f"spectra/illuminant/interpolated/{name}.csv"


def _load_pipeline_yaml() -> dict:
    path = REPO / "config" / "pipeline.yaml"
    return yaml.safe_load(path.read_text()) if path.is_file() else {}


def _camera_config_defaults(camera_name: str) -> dict:
    """Return merged camera model for a recipe, with safe fallbacks."""
    recipe = RECIPES_DIR / f"{camera_name}.yaml"
    if not recipe.is_file():
        return {}
    try:
        return load_camera_model(recipe)
    except Exception:
        return {}


def _results_data() -> dict | None:
    png_dir = OUT_DIR / "colorchecker_noisy_png"
    if not png_dir.is_dir():
        return None
    images = {}
    bust = int(time.time())
    for key, fname in [
        ("clean_rgb", "clean_rgb8.png"),
        ("noisy_rgb", "noisy_rgb8.png"),
        ("clean_demosaic", "clean_demosaic_rgb8.png"),
        ("noisy_demosaic", "noisy_demosaic_rgb8.png"),
        ("bayer_mono", "noisy_mono_16.png"),
    ]:
        p = png_dir / fname
        if p.is_file():
            images[key] = f"/static/out/colorchecker_noisy_png/{fname}?t={bust}"

    emva = None
    emva_path = OUT_DIR / "emva_validation_report.json"
    if emva_path.is_file():
        emva = json.loads(emva_path.read_text())

    demosaic = None
    demosaic_path = OUT_DIR / "demosaic_linear_metrics.json"
    if demosaic_path.is_file():
        demosaic = json.loads(demosaic_path.read_text())

    manifest = None
    manifests = sorted(OUT_DIR.glob("run_pipeline_*.json"), reverse=True)
    if manifests:
        manifest = json.loads(manifests[0].read_text())

    return {"images": images, "emva": emva, "demosaic": demosaic, "manifest": manifest}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    pipeline = _load_pipeline_yaml()
    camera_name = (pipeline.get("paths", {}) or {}).get("camera_model_name", "iphone_8")
    groups = _grouped_cameras()
    illuminants = _illuminants()
    lensfiles = _lensfiles()
    cam_cfg = _camera_config_defaults(camera_name)
    results = _results_data()
    return templates.TemplateResponse(request, "base.html", {
        "groups": groups,
        "illuminants": illuminants,
        "lensfiles": lensfiles,
        "pipeline": pipeline,
        "camera_name": camera_name,
        "cam_cfg": cam_cfg,
        "results": results,
    })


@app.get("/cameras", response_class=HTMLResponse)
async def camera_list(request: Request, q: str = ""):
    groups = _grouped_cameras(q)
    return templates.TemplateResponse(request, "camera_list.html", {
        "groups": groups,
        "query": q,
    })


@app.get("/cameras/{name}/config-form", response_class=HTMLResponse)
async def config_form(request: Request, name: str):
    pipeline = _load_pipeline_yaml()
    illuminants = _illuminants()
    lensfiles = _lensfiles()
    cam_cfg = _camera_config_defaults(name)
    return templates.TemplateResponse(request, "config_form.html", {
        "pipeline": pipeline,
        "illuminants": illuminants,
        "lensfiles": lensfiles,
        "camera_name": name,
        "cam_cfg": cam_cfg,
    })


@app.post("/run")
async def run_pipeline(
    request: Request,
    camera_name: str = Form(...),
    xres: int = Form(960),
    yres: int = Form(640),
    pixelsamples: int = Form(64),
    illuminant: str = Form("D65"),
    light_scale: float = Form(2.0),
    cam_dist: float = Form(4.25),
    film: str = Form("rgb"),
    lens_type_override: str = Form(""),
    aperture_mm: str = Form(""),
    focus_distance: str = Form(""),
    fov_deg: str = Form(""),
    lensfile: str = Form(""),
    sf_mode: str = Form("analytic"),
    target_lux: str = Form("200"),
    exposure_time: float = Form(0.06),
    noise_seed: int = Form(0),
    exposure_scale: str = Form(""),
    wb_enabled: str = Form("false"),
    ccm_enabled: str = Form("false"),
    validate_cc: str = Form("true"),
    validate_emva: str = Form("true"),
    validate_demosaic: str = Form("true"),
    gpu_enabled: str = Form("false"),
    calibration_policy: str = Form("semi_strict"),
    strict_qe: str = Form("false"),
    spectral_lambda_min: float = Form(360.0),
    spectral_lambda_max: float = Form(830.0),
    dry_run: str = Form("false"),
):
    def _bool(v: str) -> bool:
        return v.lower() in ("true", "1", "on", "yes")

    def _nullable_float(v: str):
        v = v.strip()
        return float(v) if v else None

    cfg: dict = {
        "schema_version": 1,
        "paths": {
            "scene_builder": "tools/build_colorchecker_scene.py",
            "pbrt": "third_party/pbrt-v4/build/pbrt",
            "noise_tool": "tools/apply_emva_noise.py",
            "sensor_forward_tool": "tools/spectral_sensor_forward.py",
            "pbrt_exr_to_electrons_tool": "tools/pbrt_spectral_exr_to_electrons.py",
            "validate_tool": "tools/validate_colorchecker.py",
            "validate_demosaic_tool": "tools/validate_demosaic_linear.py",
            "validate_emva_tool": "tools/validate_emva_model.py",
            "psf_tool": "tools/apply_spectral_psf.py",
            "emva_validation_report": "out/emva_validation_report.json",
            "scene_file": "scenes/generated/colorchecker.pbrt",
            "exr_out": "out/colorchecker_spectral.exr",
            "camera_model_name": camera_name,
            "sensor_forward_electrons_npz": "out/sensor_forward_electrons.npz",
            "demosaic_metrics_json": "out/demosaic_linear_metrics.json",
            "out_dir": "out",
        },
        "render": {
            "light_scale": light_scale,
            "cam_dist": cam_dist,
            "xres": xres,
            "yres": yres,
            "pixelsamples": pixelsamples,
            "film": film,
            "film_output": None,
            "spectral_nbuckets": 16,
            "spectral_lambda_min": spectral_lambda_min,
            "spectral_lambda_max": spectral_lambda_max,
            "illuminant": _illuminant_path(illuminant),
            "builder_extra_args": [],
            "gpu_enabled": _bool(gpu_enabled),
            "pbrt_args": ["--stats"],
        },
        "validate": {"enabled": _bool(validate_cc)},
        "validate_emva": {"enabled": _bool(validate_emva)},
        "lens_type_override": lens_type_override.strip() or None,
        "lens_overrides": {
            "pinhole_fov_deg": _nullable_float(fov_deg),
            "thinlens_fov_deg": _nullable_float(fov_deg),
            "thinlens_lens_radius": None,
            "thinlens_focal_distance": _nullable_float(focus_distance),
            "realistic_lensfile": lensfile.strip() or None,
            "realistic_aperture_diameter_mm": _nullable_float(aperture_mm),
            "realistic_focus_distance": _nullable_float(focus_distance),
        },
        "realistic_focus_distance_override": _nullable_float(focus_distance) or 4.0,
        "exposure_time_override_s": exposure_time,
        "sensor_forward": {
            "mode": sf_mode,
            "enabled": True,
            "target_illuminance_lux": _nullable_float(target_lux),
        },
        "noise": {
            "enabled": True,
            "seed": noise_seed,
            "exposure_scale": _nullable_float(exposure_scale),
            "preview_percentile": 99.5,
            "preview_no_normalize": True,
            "preview_white_balance_enabled": _bool(wb_enabled),
            "preview_color_correction_enabled": _bool(ccm_enabled),
        },
        "validate_demosaic": {
            "enabled": _bool(validate_demosaic),
            "crop": 2,
        },
        "strict_physical_accuracy": {
            "strict_qe_validation": _bool(strict_qe),
            "strict_calibration_validation": False,
            "calibration_tier_policy": calibration_policy,
        },
    }

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", dir=str(OUT_DIR), delete=False, prefix="run_web_"
    )
    yaml.dump(cfg, tmp, default_flow_style=False)
    tmp.close()
    cfg_path = Path(tmp.name)

    cmd = [sys.executable, str(REPO / "tools" / "run_pipeline.py"), "--config", str(cfg_path)]
    if _bool(dry_run):
        cmd.append("--dry-run")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"log": [], "done": False, "error": False}

    async def _run():
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO),
        )
        _jobs[job_id]["proc"] = proc
        async for line in proc.stdout:
            _jobs[job_id]["log"].append(line.decode(errors="replace").rstrip())
        await proc.wait()
        _jobs[job_id]["done"] = True
        _jobs[job_id]["error"] = proc.returncode != 0
        try:
            cfg_path.unlink()
        except OSError:
            pass

    asyncio.create_task(_run())
    return JSONResponse({"job_id": job_id})


@app.get("/run/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    async def _events() -> AsyncGenerator[str, None]:
        sent = 0
        while True:
            if await request.is_disconnected():
                break
            job = _jobs.get(job_id)
            if job is None:
                yield "data: [job not found]\n\n"
                break
            log = job["log"]
            while sent < len(log):
                line = log[sent].replace("\n", " ")
                yield f"data: {line}\n\n"
                sent += 1
            if job["done"]:
                status = "error" if job["error"] else "done"
                yield f"event: complete\ndata: {status}\n\n"
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(_events(), media_type="text/event-stream")


@app.get("/run/{job_id}/results", response_class=HTMLResponse)
async def job_results(request: Request, job_id: str):
    results = _results_data()
    return templates.TemplateResponse(request, "results.html", {
        "results": results,
    })
