"""Geometry regressions with known walls and explicit contact evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from droplet_analysis.core import Settings, Wall
from droplet_analysis.v5 import Analyzer
from droplet_analysis.optical import complete_confirmed_wall_boundary, limit_valid_view


class GeometryTests(unittest.TestCase):
    def test_oblique_wall_reaches_bottom_left_of_old_search_window(self):
        y, x = np.indices((480, 640))
        truth = 440-.65*y
        background = np.where(x >= truth, 35, 255).astype(np.uint8)
        wall = Wall(background, Settings())
        self.assertGreater(wall.y_max, 465)
        self.assertLess(wall.points[-1, 0], .32*640)
        np.testing.assert_allclose(wall.points[15:-15, 0], 440-.65*wall.points[15:-15, 1], atol=1.2)
        self.assertTrue(np.all(wall.normals[:, 0] < 0))
        self.assertLess(np.max(np.abs(np.linalg.norm(wall.normals, axis=1)-1)), 1e-5)
        rows = [dict(frame=n, phase='postimpact', bbox=[100, 240, 200, bottom], measurement_status='valid', flags=[])
                for n, bottom in enumerate([300, 400, 470])]
        window = limit_valid_view(rows, wall, [0, 0, 640, 468])
        self.assertEqual(window['first_excluded_frame'], 2)

    def test_short_distractor_does_not_replace_wall(self):
        y, x = np.indices((480, 640))
        truth = 440-.65*y
        background = np.where(x >= truth, 80, 255).astype(np.uint8)
        background[30:85, 60:80] = 0
        wall = Wall(background, Settings())
        self.assertGreater(wall.y_max, 465)
        np.testing.assert_allclose(wall.points[15:-15, 0], 440-.65*wall.points[15:-15, 1], atol=1.3)

    def test_search_bounds_are_validated(self):
        background = np.full((240, 400), 255, np.uint8)
        background[:, 200:] = 30
        for bounds in [[0, 0, 401, 240], [0, 0, 400], [0, 0, 400.5, 240], [200, 0, 205, 240]]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                Wall(background, Settings(wall_search_roi=bounds))

    def test_manual_wall_has_no_extrapolated_solid_outside_reference(self):
        background = np.full((240, 400), 255, np.uint8)
        wall = Wall(background, Settings(wall_points=[[200, 50], [190, 100], [180, 150], [170, 190]]))
        self.assertFalse(wall.solid[:50].any())
        self.assertFalse(wall.solid[191:].any())
        self.assertTrue(wall.solid[120, 300])
        self.assertTrue(np.isfinite(wall.distance).all())

    def make_contact(self):
        y, x = np.indices((240, 400))
        background = np.where(x >= 200, 35, 255).astype(np.uint8)
        analyzer = Analyzer([background]*3, Settings(register=False))
        self.assertEqual(analyzer.roi, [0, 0, 400, 228])
        full = (((x-180)/30)**2+((y-120)/25)**2 <= 1) & analyzer.allowed
        missing = full & (analyzer.wall.distance < 3) & (y > 112) & (y < 128)
        mask = full & ~missing
        result = dict(contact_segments=[dict(s_start_px=85., s_end_px=135.)])
        return analyzer, full, mask, missing, result

    def test_confirmed_enclosed_rim_is_completed_without_mutating_evidence(self):
        analyzer, full, mask, missing, result = self.make_contact()
        analyzer.contact_confirmed = True
        original = mask.copy()
        revised, added = complete_confirmed_wall_boundary(analyzer, mask, result)
        self.assertTrue(added.any())
        self.assertTrue(np.all(revised[missing]))
        self.assertFalse((added & analyzer.wall.solid).any())
        self.assertFalse((added & (analyzer.wall.distance > 7)).any())
        np.testing.assert_array_equal(original, mask)
        np.testing.assert_array_equal(revised, full)

    def test_unconfirmed_or_open_air_gap_is_not_closed(self):
        analyzer, _, mask, _, result = self.make_contact()
        revised, added = complete_confirmed_wall_boundary(analyzer, mask, result)
        self.assertFalse(added.any())
        np.testing.assert_array_equal(revised, mask)
        analyzer.contact_confirmed = True
        mask[118:122, :] = False  # Resolved gap opens to the free exterior.
        _, added = complete_confirmed_wall_boundary(analyzer, mask, result)
        self.assertFalse(added.any())

    def test_missing_frame_resets_contact_confirmation(self):
        analyzer, full, mask, _, _ = self.make_contact()
        image = analyzer.background.astype(np.uint8).copy()
        image[full] = 60
        for number in [10, 11, 12]:
            result, _, _, _ = analyzer.measure_main(mask, image, number)
        self.assertTrue(analyzer.contact_confirmed)
        result, _, _, _ = analyzer.measure_main(mask, image, 14)
        self.assertFalse(analyzer.contact_confirmed)
        _, added = complete_confirmed_wall_boundary(analyzer, mask, result)
        self.assertFalse(added.any())

    def test_current_gray_neck_connects_wall_fragments_but_white_gap_does_not(self):
        y,x=np.indices((240,400))
        background=np.where(x>=200,35,255).astype(np.uint8)
        liquid=np.zeros_like(background,dtype=bool)
        liquid[120:181,170:200]=True
        liquid[70:100,192:200]=True
        for gray_neck in [False,True]:
            analyzer=Analyzer([background]*3,Settings(register=False))
            image=background.copy();image[liquid]=60
            if gray_neck:image[99:122,198:200]=80
            main,satellites,_=analyzer.main_component(liquid,50,image)
            with self.subTest(gray_neck=gray_neck):
                self.assertEqual(bool(main[80,195]),gray_neck)
                self.assertEqual(analyzer.last_gray_neck_bridge_area>0,gray_neck)
                if not gray_neck:self.assertTrue(satellites)

    def test_gray_neck_does_not_expand_a_compact_first_contact_drop(self):
        y,x=np.indices((240,400))
        background=np.where(x>=200,35,255).astype(np.uint8)
        analyzer=Analyzer([background]*3,Settings(register=False))
        liquid=np.zeros_like(background,dtype=bool)
        liquid[120:151,140:200]=True
        liquid[70:100,192:200]=True
        image=background.copy();image[liquid]=60;image[99:122,198:200]=80
        main,_,_=analyzer.main_component(liquid,50,image)
        self.assertFalse(main[80,195])
        self.assertEqual(analyzer.last_gray_neck_bridge_area,0)


if __name__ == '__main__':
    unittest.main()
