import copy
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.request
import urllib.error

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from watercar_tool.io import load_json, atomic_json
from watercar_tool.jobs import preflight, create_run, run
from watercar_tool.server import Project, ReviewService, Conflict, validate_edit
from watercar_tool.review import render_review, ASSET_FIELDS
from watercar_tool.datasets import entry_folder, source_info, dataset_key, readable_name


def make_sequence(folder, count=40):
    folder.mkdir(parents=True)
    for n in range(count):
        image = np.full((240, 400), 255, np.uint8); image[:, 200:] = 40
        if n >= 32:
            cy = 206 if n in [37, 38] else 150
            mask = np.zeros(image.shape, np.uint8)
            cv2.ellipse(mask, (175, cy), (30, 22), 0, 0, 360, 1, -1)
            mask[:, 200:] = 0; image[mask > 0] = 80
        ok, data = cv2.imencode(".bmp", image); assert ok
        (folder / f"camera-{n}.bmp").write_bytes(data.tobytes())


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.input = cls.base / "合成序列 路径"
        make_sequence(cls.input)
        cls.settings = load_json(ROOT / "src/droplet_analysis/config.json")["defaults"]
        cls.settings["register"] = False
        cls.folder = create_run(cls.input, cls.base / "runs", cls.settings, workers=1)
        run(cls.folder)
        cls.state = load_json(cls.folder / "run.json")
        cls.case = cls.folder / entry_folder(cls.state['sequences'][0])
        cls.manifest = load_json(cls.case / "measurements.json")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def edit(self):
        m = self.manifest
        return dict(schema_version=5, sequence=m["sequence"], analysis_signature=m["analysis_signature"],
                    coordinate_system="aligned_reference_px", impact_frame=None, frames={"33": {"note": "synthetic review", "reviewed": True}})

    def test_numerical_modules_match_release_provenance(self):
        proof = load_json(ROOT / "ENGINE_PROVENANCE.json")
        for name, sha in proof["numerical_modules_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src/droplet_analysis" / name).read_bytes()).hexdigest(), sha)

    def test_pixel_only_complete_flow(self):
        m = self.manifest
        self.assertEqual(self.state["status"], "complete")
        self.assertEqual(len(m["frames"]), 40)
        self.assertEqual(m["units"], "px")
        self.assertIsNone(m["physical_calibration"]["mm_per_px"])
        self.assertEqual(m["impact"]["first_contact_frame"], 32)
        self.assertIsNone(m["impact_point"]["s_mm"])
        self.assertIsNotNone(m["analysis_window"]["first_excluded_frame"])
        self.assertTrue((self.folder / "review.html").exists())
        for frame in m['frames']:
            self.assertTrue((self.case / frame['observed_mask']).exists())
            self.assertTrue((self.case / frame['contact_completion_mask']).exists())
            if frame['frame'] < 34:
                self.assertEqual(frame['contact_completion_area_px2'], 0)

    def test_formal_and_trace_boundaries(self):
        with (self.case / "measurements.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream); rows = list(reader); fields = reader.fieldnames
        self.assertFalse(any("_mm" in key for key in fields))
        self.assertTrue(rows)
        self.assertTrue(all(r["phase"] == "postimpact" and r["within_valid_view"] == "True" for r in rows))
        with (self.case / "all_frames_trace.csv").open(encoding="utf-8-sig", newline="") as stream:
            trace = list(csv.DictReader(stream))
        self.assertEqual(len(trace), 40)
        self.assertTrue(any(r["phase"] == "after_valid_view" for r in trace))

    def test_review_recovery_and_conflict(self):
        project = Project(self.folder); before = project.read_state()
        after = project.save_state(before["revision"], {self.manifest["analysis_signature"]: self.edit()})
        self.assertEqual(Project(self.folder).read_state(), after)
        with self.assertRaises(Conflict): project.save_state(before["revision"], {})

    def test_history_version_and_completion_asset_paths(self):
        old = copy.deepcopy(self.manifest)
        old['algorithm_version'] = '5.3.1-confirmed-contact-continuation-and-observed-front'
        self.assertIn('算法 5.3.1', render_review(old))
        project = Project(self.folder)
        view = next(iter(project.groups.values()))['view']
        for f in view['frames']:
            for key in ASSET_FIELDS:
                self.assertTrue(f[key].startswith('@asset/'))
                self.assertTrue(project.assets[f[key].removeprefix('@asset/')].exists())

    def test_wrong_signature_or_frame_rejected(self):
        value = self.edit(); value["analysis_signature"] = "wrong"
        with self.assertRaises(ValueError): validate_edit(value, self.manifest)
        value = self.edit(); value["frames"]["9999"] = {}
        with self.assertRaises(ValueError): validate_edit(value, self.manifest)

    def test_review_batch_validation_no_partial_write(self):
        project = Project(self.folder); before = project.read_state()
        with self.assertRaises(ValueError): project.save_state(before["revision"], {"unknown": self.edit()})
        self.assertEqual(project.read_state(), before)

    def test_resume_does_not_rewrite_completed_group(self):
        p = self.case / "measurements.json"; before = p.read_bytes()
        run(self.folder)
        self.assertEqual(p.read_bytes(), before)

    def test_verified_alignment_cache_reproduces_fresh_geometry(self):
        from watercar_tool.engine import analyze_sequence
        output=self.base/'cached_geometry'
        analyze_sequence(self.input,output,self.settings,alignment_cache=self.case,sequence_name=self.manifest['sequence'])
        cached=load_json(output/'measurements.json')
        self.assertEqual(cached['frames'],self.manifest['frames'])
        self.assertEqual(cached['impact'],self.manifest['impact'])
        self.assertEqual(cached['analysis_window'],self.manifest['analysis_window'])
        self.assertTrue(cached['alignment_cache']['raw_hashes_reverified'])

    def test_resume_recovers_promoted_group(self):
        path = self.folder / "run.json"
        state = load_json(path); original = copy.deepcopy(state)
        state["completed"] = []; atomic_json(path, state)
        try:
            run(self.folder)
            self.assertEqual(load_json(path)["completed"], original["completed"])
            self.assertEqual(load_json(path)["status"], "complete")
        finally:
            atomic_json(path, original)

    def test_local_service_and_restart(self):
        service = ReviewService()
        try:
            url = service.open_project(self.folder)
            html = urllib.request.urlopen(url).read().decode()
            self.assertNotIn("__PERSISTENCE_UI__", html)
            self.assertIn("const toolContext=", html)
            api = url.rsplit("/", 1)[0] + "/api/reviews"
            value = json.load(urllib.request.urlopen(api))
            self.assertIn("revision", value)
            bad = urllib.request.Request(api, data=json.dumps(dict(revision=value["revision"], changes={})).encode(), headers={"Content-Type":"application/json", "Origin":"https://example.invalid"})
            with self.assertRaises(urllib.error.HTTPError): urllib.request.urlopen(bad)
        finally: service.close()

    def test_saved_export_keeps_cutoff_and_original(self):
        project = Project(self.folder); before = (self.case / "measurements.json").read_bytes()
        final = copy.deepcopy(self.manifest)
        dest = Path(project.export(final["analysis_signature"], project.read_state()["revision"], final))
        self.assertTrue((dest / "thickness_vs_frame.png").exists())
        self.assertEqual((self.case / "measurements.json").read_bytes(), before)
        self.assertEqual(load_json(dest / "measurements.json")["analysis_window"], self.manifest["analysis_window"])
        exported = load_json(dest / 'measurements.json')
        for f in exported['frames']:
            for key in ASSET_FIELDS:
                self.assertTrue((dest / f[key]).exists())

    def test_bad_input_reported(self):
        short = self.base / "short"; make_sequence(short, 2)
        with self.assertRaisesRegex(ValueError, "32"):
            preflight(short, self.settings)

    def test_same_named_nested_sequences_distinct(self):
        parent = self.base / "two"
        make_sequence(parent / "one" / "same", 32); make_sequence(parent / "two" / "same", 32)
        info = preflight(parent, self.settings)
        self.assertEqual(len({x["id"] for x in info["sequences"]}), 2)
        self.assertEqual(len({x["label"] for x in info["sequences"]}), 2)

    def test_readable_provenance_survives_resume(self):
        entry=self.state['sequences'][0]
        self.assertIn('合成序列',self.folder.name)
        self.assertIn('合成序列',self.case.name)
        info=load_json(self.case/'source_info.json')
        self.assertEqual(info['source_directory'],str(self.input))
        self.assertEqual(info['output_directory'],self.case.name)
        with (self.folder/'datasets.csv').open(encoding='utf-8-sig',newline='') as stream:
            row=next(csv.DictReader(stream))
        self.assertEqual(row['source_directory'],str(self.input))
        self.assertEqual(row['status'],'complete')
        self.assertEqual(entry_folder({'id':'legacy'}),'legacy')

    def test_legacy_id_directory_can_still_resume(self):
        legacy=self.base/'legacy_run';shutil.copytree(self.folder,legacy)
        state=load_json(legacy/'run.json');entry=state['sequences'][0]
        (legacy/entry.pop('output_directory')).rename(legacy/entry['id'])
        for key in ['display_label','source_info']:entry.pop(key,None)
        atomic_json(legacy/'run.json',state)
        path=legacy/entry['id']/'measurements.json';before=path.read_bytes()
        run(legacy)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(load_json(legacy/'run.json')['status'],'complete')

    def test_condition_path_extraction_and_numeric_sort(self):
        paths=[self.base/'实验'/angle/'侧面'/condition/'BMP20261014'
               for angle,condition in [('75°','2.0mm_100mm_001'),('65°','2.0mm_1000mm_001'),
                                       ('65°','2.0mm_200mm_010'),('65°','2.0mm_200mm_002')]]
        infos=[source_info(p,self.base) for p in paths]
        self.assertEqual(infos[0]['display_label'],'75° / 侧面 / 2.0mm_100mm_001')
        self.assertEqual(infos[0]['acquisition'],'BMP20261014')
        ordered=sorted([{'source_info':i} for i in infos],key=dataset_key)
        self.assertEqual([x['source_info']['condition'] for x in ordered],
                         ['2.0mm_200mm_002','2.0mm_200mm_010','2.0mm_1000mm_001','2.0mm_100mm_001'])
        self.assertNotRegex(readable_name('65° / 侧面 / 2.0mm_100mm_001'),r'[<>:"/\\|?*]')

    def test_multigroup_service_loads_only_selected_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            for i,condition in enumerate(['2.0mm_1000mm_001','2.0mm_200mm_002','2.0mm_100mm_010']):
                m=copy.deepcopy(self.manifest)
                m['analysis_signature']='signature-'+str(i)
                # Repeated sequence names must remain distinct by signature.
                m['sequence']='same-camera-folder'
                m['source_directory']=str(self.base/'65°'/condition/'BMP20261014')
                for frame in m['frames']:
                    for key in ASSET_FIELDS: frame[key]=str((self.case/frame[key]).resolve())
                destination=folder/str(i);destination.mkdir()
                atomic_json(destination/'measurements.json',m)
            service=ReviewService()
            try:
                url=service.open_project(folder);prefix=url.rsplit('/',1)[0]
                def payload(page):
                    return json.loads(re.search(r'<script id="manifest" type="application/json">(.*?)</script>',page,re.S)[1])
                first=payload(urllib.request.urlopen(url).read().decode())
                self.assertEqual(len(first['sequences']),1)
                self.assertEqual(first['sequences'][0]['analysis_signature'],'signature-2')
                self.assertEqual([x['analysis_signature'] for x in first['dataset_catalog']],['signature-2','signature-1','signature-0'])
                self.assertTrue(all('frames' not in x for x in first['dataset_catalog']))
                selected=payload(urllib.request.urlopen(prefix+'/signature-0/review.html').read().decode())
                self.assertEqual(selected['sequences'][0]['analysis_signature'],'signature-0')
                geometry=json.load(urllib.request.urlopen(prefix+'/api/manifests/signature-1'))
                self.assertEqual(len(geometry['frames']),40)
                self.assertTrue(geometry['frames'][0]['image'].startswith('/s/'))
                changes={}
                for signature in ['signature-1','signature-2']:
                    edit=self.edit();edit['analysis_signature']=signature;edit['sequence']='same-camera-folder';changes[signature]=edit
                request=urllib.request.Request(prefix+'/api/reviews',data=json.dumps(dict(revision=0,changes=changes)).encode(),headers={'Content-Type':'application/json'})
                saved=json.load(urllib.request.urlopen(request))
                self.assertEqual(set(saved['sequences']),set(changes))
                with self.assertRaises(urllib.error.HTTPError):urllib.request.urlopen(prefix+'/api/manifests/unknown')
            finally:service.close()


if __name__ == "__main__":
    unittest.main()
