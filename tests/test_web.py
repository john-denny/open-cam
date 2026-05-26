"""Tests for the open-cam web UI (web/main.py).

Run with:
    PYTHONPATH=tools:. uv run python -m unittest tests/test_web.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

REPO = Path(__file__).parent.parent.resolve()
OUT_DIR = REPO / "out"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "web"))

from starlette.testclient import TestClient

from web.main import (
    app,
    _brand,
    _display,
    _grouped_cameras,
    _illuminant_path,
    _illuminants,
    _lensfiles,
)

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------

class TestRequiredFilesPresent(unittest.TestCase):

    def _assert_file(self, *parts: str) -> None:
        p = REPO.joinpath(*parts)
        self.assertTrue(p.is_file(), f"Missing required file: {p}")

    def _assert_dir(self, *parts: str) -> None:
        p = REPO.joinpath(*parts)
        self.assertTrue(p.is_dir(), f"Missing required directory: {p}")

    # Web app source
    def test_main_py(self):
        self._assert_file("web", "main.py")

    def test_template_base(self):
        self._assert_file("web", "templates", "base.html")

    def test_template_camera_list(self):
        self._assert_file("web", "templates", "camera_list.html")

    def test_template_config_form(self):
        self._assert_file("web", "templates", "config_form.html")

    def test_template_results(self):
        self._assert_file("web", "templates", "results.html")

    def test_static_css(self):
        self._assert_file("web", "static", "style.css")

    # Config
    def test_pipeline_yaml(self):
        self._assert_file("config", "pipeline.yaml")

    def test_camera_recipes_dir(self):
        self._assert_dir("config", "camera_recipes")

    def test_camera_recipes_not_empty(self):
        recipes = list((REPO / "config" / "camera_recipes").glob("*.yaml"))
        self.assertGreater(len(recipes), 0, "No camera recipe YAMLs found")

    def test_lenses_dir(self):
        self._assert_dir("config", "lenses")

    def test_lenses_dat_files(self):
        dats = list((REPO / "config" / "lenses").glob("*.dat"))
        self.assertGreater(len(dats), 0, "No .dat lens files found")

    def test_illuminants_dir(self):
        self._assert_dir("spectra", "illuminant", "interpolated")

    def test_illuminants_not_empty(self):
        csvs = list((REPO / "spectra" / "illuminant" / "interpolated").glob("*.csv"))
        self.assertGreater(len(csvs), 0, "No illuminant CSVs found")

    # Known key illuminants
    def test_illuminant_d65_present(self):
        self._assert_file("spectra", "illuminant", "interpolated", "D65.csv")

    def test_illuminant_d55_present(self):
        self._assert_file("spectra", "illuminant", "interpolated", "D55.csv")

    # Known lens prescriptions
    def test_lensfile_dgauss(self):
        self._assert_file("config", "lenses", "dgauss.50mm.dat")

    def test_lensfile_wide(self):
        self._assert_file("config", "lenses", "wide_22mm.dat")

    # Pipeline tools
    def test_tool_run_pipeline(self):
        self._assert_file("tools", "run_pipeline.py")

    def test_tool_apply_emva_noise(self):
        self._assert_file("tools", "apply_emva_noise.py")

    def test_tool_camera_model(self):
        self._assert_file("tools", "camera_model.py")

    # Known camera recipes
    def test_recipe_nikon_z6(self):
        self._assert_file("config", "camera_recipes", "nikon_z6.yaml")

    def test_recipe_iphone_8(self):
        self._assert_file("config", "camera_recipes", "iphone_8.yaml")

    def test_recipe_default(self):
        self._assert_file("config", "camera_recipes", "default.yaml")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestBrandHelper(unittest.TestCase):

    def test_canon(self):
        self.assertEqual(_brand("canon_eos_80d"), "Canon")

    def test_nikon(self):
        self.assertEqual(_brand("nikon_z6"), "Nikon")

    def test_sony(self):
        self.assertEqual(_brand("sony_alpha_6300"), "Sony")

    def test_fujifilm(self):
        self.assertEqual(_brand("fujifilm_x_t4"), "Fujifilm")

    def test_apple_iphone(self):
        self.assertEqual(_brand("iphone_8"), "Apple")

    def test_research(self):
        self.assertEqual(_brand("research_thinlens_normal"), "Research")

    def test_study_maps_to_research(self):
        self.assertEqual(_brand("study_nikon_z6_minvar"), "Research")

    def test_default(self):
        self.assertEqual(_brand("default"), "Default")

    def test_unknown_returns_other(self):
        self.assertEqual(_brand("zzz_unknown_camera"), "Other")


class TestDisplayHelper(unittest.TestCase):

    def test_underscores_to_spaces(self):
        self.assertEqual(_display("nikon_z6"), "Nikon Z6")

    def test_title_case(self):
        self.assertEqual(_display("canon_eos_80d"), "Canon Eos 80D")

    def test_already_clean(self):
        self.assertEqual(_display("sigma"), "Sigma")


class TestGroupedCameras(unittest.TestCase):

    def test_returns_list_of_groups(self):
        groups = _grouped_cameras()
        self.assertIsInstance(groups, list)
        self.assertGreater(len(groups), 0)

    def test_each_group_has_brand_and_cameras(self):
        for group in _grouped_cameras():
            self.assertIn("brand", group)
            self.assertIn("cameras", group)
            self.assertIsInstance(group["cameras"], list)

    def test_each_camera_has_name_and_display(self):
        for group in _grouped_cameras():
            for cam in group["cameras"]:
                self.assertIn("name", cam)
                self.assertIn("display", cam)

    def test_nikon_group_present(self):
        brands = [g["brand"] for g in _grouped_cameras()]
        self.assertIn("Nikon", brands)

    def test_query_filters_results(self):
        groups = _grouped_cameras("nikon")
        all_names = [c["name"] for g in groups for c in g["cameras"]]
        self.assertTrue(all("nikon" in n for n in all_names))

    def test_query_no_match_returns_empty(self):
        groups = _grouped_cameras("zzznomatch")
        self.assertEqual(groups, [])

    def test_brand_order_respected(self):
        groups = _grouped_cameras()
        brands = [g["brand"] for g in groups]
        canon_idx = brands.index("Canon") if "Canon" in brands else -1
        nikon_idx = brands.index("Nikon") if "Nikon" in brands else -1
        if canon_idx >= 0 and nikon_idx >= 0:
            self.assertLess(canon_idx, nikon_idx)

    def test_index_yaml_excluded(self):
        all_names = [c["name"] for g in _grouped_cameras() for c in g["cameras"]]
        self.assertNotIn("INDEX", all_names)


class TestIlluminants(unittest.TestCase):

    def test_returns_list(self):
        result = _illuminants()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_d65_present(self):
        self.assertIn("D65", _illuminants())

    def test_d55_present(self):
        self.assertIn("D55", _illuminants())

    def test_sorted(self):
        result = _illuminants()
        self.assertEqual(result, sorted(result))

    def test_no_extensions(self):
        for name in _illuminants():
            self.assertFalse(name.endswith(".csv"), f"Expected stem, got: {name}")


class TestIlluminantPath(unittest.TestCase):

    def test_builds_correct_path(self):
        self.assertEqual(
            _illuminant_path("D65"),
            "spectra/illuminant/interpolated/D65.csv",
        )

    def test_preserves_name(self):
        self.assertIn("F11", _illuminant_path("F11"))


class TestLensfiles(unittest.TestCase):

    def test_returns_list(self):
        result = _lensfiles()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    def test_each_entry_has_required_keys(self):
        for lf in _lensfiles():
            self.assertIn("path", lf)
            self.assertIn("label", lf)
            self.assertIn("stem", lf)

    def test_path_has_dat_extension(self):
        for lf in _lensfiles():
            self.assertTrue(lf["path"].endswith(".dat"), lf["path"])

    def test_path_prefixed_correctly(self):
        for lf in _lensfiles():
            self.assertTrue(lf["path"].startswith("config/lenses/"), lf["path"])

    def test_dgauss_present(self):
        stems = [lf["stem"] for lf in _lensfiles()]
        self.assertIn("dgauss.50mm", stems)

    def test_wide_present(self):
        stems = [lf["stem"] for lf in _lensfiles()]
        self.assertIn("wide_22mm", stems)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

class TestAPIIndex(unittest.TestCase):

    def test_returns_200(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)

    def test_contains_shell_structure(self):
        r = client.get("/")
        self.assertIn('id="shell"', r.text)
        self.assertIn('id="sidebar"', r.text)
        self.assertIn('id="config-panel"', r.text)
        self.assertIn('id="results-panel"', r.text)

    def test_contains_htmx(self):
        r = client.get("/")
        self.assertIn("htmx.org", r.text)

    def test_contains_camera_list(self):
        r = client.get("/")
        self.assertIn("brand-group", r.text)

    def test_title(self):
        r = client.get("/")
        self.assertIn("open-cam", r.text)


class TestAPICameraList(unittest.TestCase):

    def test_returns_200(self):
        r = client.get("/cameras")
        self.assertEqual(r.status_code, 200)

    def test_contains_brand_groups(self):
        r = client.get("/cameras")
        self.assertIn("brand-group", r.text)

    def test_query_filter_nikon(self):
        r = client.get("/cameras?q=nikon")
        self.assertEqual(r.status_code, 200)
        self.assertIn("nikon", r.text.lower())
        self.assertNotIn("canon", r.text.lower())

    def test_query_no_match(self):
        r = client.get("/cameras?q=zzznomatch")
        self.assertEqual(r.status_code, 200)
        self.assertIn("No cameras match", r.text)

    def test_query_case_insensitive(self):
        lower = client.get("/cameras?q=nikon")
        upper = client.get("/cameras?q=Nikon")
        self.assertEqual(lower.text, upper.text)


class TestAPIConfigForm(unittest.TestCase):

    def test_nikon_z6_returns_200(self):
        r = client.get("/cameras/nikon_z6/config-form")
        self.assertEqual(r.status_code, 200)

    def test_iphone_8_returns_200(self):
        r = client.get("/cameras/iphone_8/config-form")
        self.assertEqual(r.status_code, 200)

    def test_form_fields_present(self):
        r = client.get("/cameras/nikon_z6/config-form")
        for field in ["xres", "yres", "pixelsamples", "light_scale", "cam_dist",
                      "film", "illuminant", "lens_type_override", "sf_mode",
                      "target_lux", "exposure_time", "noise_seed"]:
            self.assertIn(f'name="{field}"', r.text, f"Missing field: {field}")

    def test_realistic_camera_shows_lensfile_field(self):
        r = client.get("/cameras/research_realistic_wide22/config-form")
        self.assertIn('id="field-lensfile"', r.text)

    def test_realistic_camera_has_model_type_attribute(self):
        r = client.get("/cameras/research_realistic_wide22/config-form")
        self.assertIn('data-model-type="realistic"', r.text)

    def test_lensfile_options_populated(self):
        r = client.get("/cameras/research_realistic_wide22/config-form")
        self.assertIn("dgauss", r.text.lower())
        self.assertIn("wide", r.text.lower())

    def test_illuminant_options_populated(self):
        r = client.get("/cameras/nikon_z6/config-form")
        self.assertIn("D65", r.text)
        self.assertIn("D55", r.text)

    def test_tooltips_present(self):
        r = client.get("/cameras/nikon_z6/config-form")
        self.assertIn("tip-icon", r.text)
        self.assertIn("data-tip", r.text)

    def test_unknown_camera_returns_200(self):
        # Should degrade gracefully — empty cam_cfg, form still renders
        r = client.get("/cameras/does_not_exist/config-form")
        self.assertEqual(r.status_code, 200)


class TestAPIRun(unittest.TestCase):

    _BASE_FORM = {
        "camera_name": "nikon_z6",
        "xres": "64",
        "yres": "64",
        "pixelsamples": "1",
        "illuminant": "D65",
        "light_scale": "1.0",
        "cam_dist": "4.0",
        "film": "spectral",
        "lens_type_override": "",
        "aperture_mm": "",
        "focus_distance": "",
        "fov_deg": "",
        "lensfile": "",
        "sf_mode": "analytic",
        "target_lux": "200",
        "exposure_time": "0.06",
        "noise_seed": "0",
        "exposure_scale": "",
        "wb_enabled": "false",
        "ccm_enabled": "false",
        "validate_cc": "true",
        "validate_emva": "false",
        "validate_demosaic": "false",
        "gpu_enabled": "false",
        "calibration_policy": "semi_strict",
        "strict_qe": "false",
        "spectral_lambda_min": "360.0",
        "spectral_lambda_max": "830.0",
        "dry_run": "true",
    }

    def test_returns_200(self):
        r = client.post("/run", data=self._BASE_FORM)
        self.assertEqual(r.status_code, 200)

    def test_returns_job_id(self):
        r = client.post("/run", data=self._BASE_FORM)
        body = r.json()
        self.assertIn("job_id", body)
        self.assertIsInstance(body["job_id"], str)
        self.assertGreater(len(body["job_id"]), 0)

    def test_missing_camera_name_rejected(self):
        form = {k: v for k, v in self._BASE_FORM.items() if k != "camera_name"}
        r = client.post("/run", data=form)
        self.assertEqual(r.status_code, 422)

    def test_each_run_gets_unique_job_id(self):
        r1 = client.post("/run", data=self._BASE_FORM)
        r2 = client.post("/run", data=self._BASE_FORM)
        self.assertNotEqual(r1.json()["job_id"], r2.json()["job_id"])


class TestAPIResults(unittest.TestCase):

    def test_results_endpoint_returns_200(self):
        r = client.post("/run", data=TestAPIRun._BASE_FORM)
        job_id = r.json()["job_id"]
        r2 = client.get(f"/run/{job_id}/results")
        self.assertEqual(r2.status_code, 200)

    def test_unknown_job_results_still_200(self):
        # Results come from disk, not job state — always returns current out/ contents
        r = client.get("/run/doesnotexist/results")
        self.assertEqual(r.status_code, 200)


class TestAPIStaticFiles(unittest.TestCase):

    def test_css_served(self):
        r = client.get("/static/style.css")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/css", r.headers.get("content-type", ""))

    def test_css_contains_variables(self):
        r = client.get("/static/style.css")
        self.assertIn("--bg", r.text)
        self.assertIn("--accent", r.text)


class TestAPIRunYAMLContents(unittest.TestCase):
    """Verify the YAML written to disk by /run matches submitted form values.

    asyncio.create_task is patched out so the subprocess never starts, leaving
    the temp file on disk for inspection. The helper cleans it up regardless.
    """

    # A form with deliberately non-default values so every field is meaningful.
    _FORM = {
        "camera_name": "nikon_z6",
        "xres": "320",
        "yres": "240",
        "pixelsamples": "8",
        "illuminant": "D55",
        "light_scale": "1.5",
        "cam_dist": "3.0",
        "film": "spectral",
        "lens_type_override": "pinhole",
        "aperture_mm": "",
        "focus_distance": "5.0",
        "fov_deg": "50",
        "lensfile": "",
        "sf_mode": "pbrt_exr",
        "target_lux": "500",
        "exposure_time": "0.1",
        "noise_seed": "42",
        "exposure_scale": "",
        "wb_enabled": "true",
        "ccm_enabled": "false",
        "validate_cc": "false",
        "validate_emva": "false",
        "validate_demosaic": "true",
        "gpu_enabled": "false",
        "calibration_policy": "research",
        "strict_qe": "true",
        "spectral_lambda_min": "380.0",
        "spectral_lambda_max": "780.0",
        "dry_run": "true",
    }

    def _post_and_read_yaml(self, form: dict) -> dict:
        """POST /run with create_task stubbed out, return parsed YAML."""
        with patch("web.main.asyncio.create_task"):
            r = client.post("/run", data=form)
        self.assertEqual(r.status_code, 200)

        # The temp file is the newest run_web_*.yaml in out/
        yaml_files = sorted(
            OUT_DIR.glob("run_web_*.yaml"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        self.assertTrue(yaml_files, "No temp YAML was written to out/")
        cfg_path = yaml_files[0]
        try:
            return yaml.safe_load(cfg_path.read_text())
        finally:
            cfg_path.unlink(missing_ok=True)

    # ── paths ────────────────────────────────────────────

    def test_camera_name(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["paths"]["camera_model_name"], "nikon_z6")

    def test_out_dir(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["paths"]["out_dir"], "out")

    # ── render ───────────────────────────────────────────

    def test_resolution(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["render"]["xres"], 320)
        self.assertEqual(cfg["render"]["yres"], 240)

    def test_pixelsamples(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["render"]["pixelsamples"], 8)

    def test_light_scale(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["render"]["light_scale"], 1.5)

    def test_cam_dist(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["render"]["cam_dist"], 3.0)

    def test_film(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["render"]["film"], "spectral")

    def test_illuminant_path(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertIn("D55", cfg["render"]["illuminant"])
        self.assertIn(".csv", cfg["render"]["illuminant"])

    def test_spectral_lambda_range(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["render"]["spectral_lambda_min"], 380.0)
        self.assertAlmostEqual(cfg["render"]["spectral_lambda_max"], 780.0)

    def test_gpu_disabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertFalse(cfg["render"]["gpu_enabled"])

    # ── lens ─────────────────────────────────────────────

    def test_lens_type_override(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["lens_type_override"], "pinhole")

    def test_fov_override(self):
        cfg = self._post_and_read_yaml(self._FORM)
        lo = cfg["lens_overrides"]
        self.assertAlmostEqual(lo["pinhole_fov_deg"], 50.0)
        self.assertAlmostEqual(lo["thinlens_fov_deg"], 50.0)

    def test_focus_distance_override(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["realistic_focus_distance_override"], 5.0)

    def test_blank_aperture_is_null(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertIsNone(cfg["lens_overrides"]["realistic_aperture_diameter_mm"])

    def test_blank_lensfile_is_null(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertIsNone(cfg["lens_overrides"]["realistic_lensfile"])

    # ── sensor forward ────────────────────────────────────

    def test_sf_mode(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["sensor_forward"]["mode"], "pbrt_exr")

    def test_sf_enabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertTrue(cfg["sensor_forward"]["enabled"])

    def test_target_lux(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["sensor_forward"]["target_illuminance_lux"], 500.0)

    def test_exposure_time(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertAlmostEqual(cfg["exposure_time_override_s"], 0.1)

    # ── noise ─────────────────────────────────────────────

    def test_noise_seed(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["noise"]["seed"], 42)

    def test_wb_enabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertTrue(cfg["noise"]["preview_white_balance_enabled"])

    def test_ccm_disabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertFalse(cfg["noise"]["preview_color_correction_enabled"])

    def test_blank_exposure_scale_is_null(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertIsNone(cfg["noise"]["exposure_scale"])

    # ── validation ────────────────────────────────────────

    def test_validate_cc_disabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertFalse(cfg["validate"]["enabled"])

    def test_validate_emva_disabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertFalse(cfg["validate_emva"]["enabled"])

    def test_validate_demosaic_enabled(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertTrue(cfg["validate_demosaic"]["enabled"])

    # ── strict / advanced ─────────────────────────────────

    def test_calibration_policy(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(
            cfg["strict_physical_accuracy"]["calibration_tier_policy"], "research"
        )

    def test_strict_qe(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertTrue(cfg["strict_physical_accuracy"]["strict_qe_validation"])

    # ── lensfile override ─────────────────────────────────

    def test_lensfile_override_written(self):
        form = {**self._FORM, "lensfile": "config/lenses/dgauss.50mm.dat"}
        cfg = self._post_and_read_yaml(form)
        self.assertEqual(
            cfg["lens_overrides"]["realistic_lensfile"],
            "config/lenses/dgauss.50mm.dat",
        )

    # ── schema ────────────────────────────────────────────

    def test_schema_version(self):
        cfg = self._post_and_read_yaml(self._FORM)
        self.assertEqual(cfg["schema_version"], 1)

    def test_all_required_top_level_keys_present(self):
        cfg = self._post_and_read_yaml(self._FORM)
        for key in ("paths", "render", "noise", "sensor_forward",
                    "validate", "validate_emva", "validate_demosaic",
                    "lens_overrides", "strict_physical_accuracy"):
            self.assertIn(key, cfg, f"Missing top-level key: {key}")


if __name__ == "__main__":
    unittest.main()
