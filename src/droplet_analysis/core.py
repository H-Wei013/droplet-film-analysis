from dataclasses import dataclass, fields
from pathlib import Path
import json
import re

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import UnivariateSpline
from .silhouette import complete_bright_cavities


@dataclass
class Settings:
    reference_frames: int = 32
    dark_seed: float = 18.0
    dark_support: float = 6.0
    edge_contrast_fraction: float = 0.25
    edge_contrast_radius_px: int = 7
    contact_seed_fraction: float = 0.5
    closing_radius: int = 1
    highlight_gap_radius: int = 10
    cavity_boundary_fraction: float = .75
    temporal_anchor_radius: int = 12
    temporal_boundary_tolerance_px: float = 2.0
    temporal_evidence_distance_px: float = 24.0
    min_component_area: int = 12
    wall_uncertainty_px: float = 2.0
    min_contact_length_px: float = 4.0
    min_film_thickness_px: float = 4.0
    normal_step_px: float = 0.5
    max_normal_distance_px: float = 220.0
    persistence_frames: int = 3
    edge_margin_px: int = 12
    register: bool = True
    roi: object = None
    wall_points: object = None
    review_frames: object = None

    @classmethod
    def from_dict(cls, value):
        allowed = {f.name for f in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f'Unknown settings: {sorted(unknown)}')
        result = cls(**value)
        if not (0 < result.dark_support <= result.dark_seed < 255):
            raise ValueError('Require 0 < dark_support <= dark_seed < 255')
        if not (0 < result.edge_contrast_fraction <= result.contact_seed_fraction <= 1):
            raise ValueError('Require 0 < edge_contrast_fraction <= contact_seed_fraction <= 1')
        if not .5 <= result.cavity_boundary_fraction <= 1:
            raise ValueError('cavity_boundary_fraction must be between .5 and 1')
        if result.reference_frames < 3 or result.persistence_frames < 1:
            raise ValueError('reference_frames >= 3 and persistence_frames >= 1 required')
        if result.temporal_anchor_radius < 1 or result.temporal_boundary_tolerance_px < 0 or result.temporal_evidence_distance_px <= 0:
            raise ValueError('Invalid temporal reconstruction distances')
        if result.normal_step_px <= 0 or result.max_normal_distance_px <= 0:
            raise ValueError('Normal sampling distances must be positive')
        if result.closing_radius < 0 or result.highlight_gap_radius < 0 or result.edge_margin_px < 0 or result.wall_uncertainty_px < 0 or result.edge_contrast_radius_px < 1:
            raise ValueError('Radii and margins must be nonnegative')
        return result


def read_gray(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    path.write_bytes(encoded.tobytes())


def frame_number(path):
    match = re.search(r'-(\d+)$', Path(path).stem)
    if not match:
        raise ValueError(f'Expected a numeric frame suffix: {path}')
    return int(match.group(1))


def discover_sequences(root):
    root = Path(root)
    dirs = sorted({p.parent for p in root.rglob('*.bmp')})
    return [d for d in dirs if list(d.glob('*.bmp'))]


def load_sequence(path):
    files = sorted(Path(path).glob('*.bmp'), key=frame_number)
    if not files:
        raise ValueError(f'No BMP frames in {path}')
    numbers = [frame_number(p) for p in files]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f'Duplicate frame numbers in {path}')
    meta_path = Path(path) / 'video_param.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    return files, numbers, meta


class Wall:
    """An optical reference edge x(y); increasing arc length follows image y.

    This is a side-view optical boundary, not a calibrated physical solid surface.
    The supported geometry is a single wall with solid on its right.
    """
    def __init__(self, background, settings):
        self.shape = background.shape
        height, width = self.shape
        smooth = cv2.GaussianBlur(background.astype(np.float32), (0, 0), 1.5)
        dx = np.gradient(smooth, axis=1)
        if settings.wall_points is not None:
            points = np.asarray(settings.wall_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
                raise ValueError('wall_points needs at least four [x, y] points')
            points = points[np.argsort(points[:, 1])]
            if np.any(np.diff(points[:, 1]) <= 0):
                raise ValueError('Wall control points must have distinct increasing y')
            if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
                raise ValueError('Wall control points must lie in the image')
            yy, xx = points[:, 1], points[:, 0]
            spline = UnivariateSpline(yy, xx, s=0)
            self.source = 'manual_reference'
        else:
            left, right = int(width * .32), int(width * .98)
            yy = np.arange(settings.edge_margin_px, height - settings.edge_margin_px)
            rows = dx[yy, left:right]
            xx = np.argmin(rows, axis=1) + left
            strength = -dx[yy, xx]
            valid = (strength > 2.0) & (xx > left + 8) & (xx < right - 8)
            runs = intervals(valid)
            if not runs:
                raise ValueError('No clear right-hand wall edge; supply wall_points')
            start, end = max(runs, key=lambda r: r[1] - r[0])
            yy, xx = yy[start:end + 1], xx[start:end + 1].astype(float)
            if len(yy) < height * .3:
                raise ValueError('Wall edge is too short; supply wall_points')
            med = ndi.median_filter(xx, size=11)
            good = np.abs(xx - med) < 5
            # Smooth the measured edge before differentiating. Fitting pixel
            # quantization tightly produced large, spurious normal rotations.
            stable = ndi.gaussian_filter1d(med.astype(float), 4, mode='nearest')
            spline = UnivariateSpline(yy[good], stable[good], s=np.count_nonzero(good) * .18)
            self.source = 'automatic_regularized_gradient_reference_v5'
        self.y_min, self.y_max = float(yy[0]), float(yy[-1])
        y_dense = np.arange(self.y_min, self.y_max + .25, .25)
        x_dense = spline(y_dense)
        arc = np.r_[0., np.cumsum(np.hypot(np.diff(x_dense), np.diff(y_dense)))]
        self.s = np.arange(0, arc[-1], 1.0)
        y = np.interp(self.s, arc, y_dense)
        x = spline(y)
        slope = spline.derivative()(y)
        norm = np.sqrt(1 + slope ** 2)
        self.points = np.c_[x, y].astype(np.float32)
        self.normals = np.c_[-1 / norm, slope / norm].astype(np.float32)
        self.x_by_y = spline(np.clip(np.arange(height), self.y_min, self.y_max))
        self.slope_by_y = spline.derivative()(np.clip(np.arange(height), self.y_min, self.y_max))
        grid_x = np.arange(width)[None, :]
        self.distance = ((self.x_by_y[:, None] - grid_x) / np.sqrt(1 + self.slope_by_y[:, None] ** 2)).astype(np.float32)
        self.solid = grid_x >= self.x_by_y[:, None]
        self.offsets = np.arange(0, settings.max_normal_distance_px + settings.normal_step_px / 2, settings.normal_step_px).astype(np.float32)
        sample = self.points[:, None, :] + self.normals[:, None, :] * self.offsets[None, :, None]
        self.map_x, self.map_y = sample[..., 0].copy(), sample[..., 1].copy()
        self.sample_valid = ((self.map_x >= 0) & (self.map_x < width - 1) & (self.map_y >= 0) & (self.map_y < height - 1))
        self.settings = settings

    def sample(self, image, nearest=False):
        return cv2.remap(image.astype(np.float32), self.map_x, self.map_y,
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def point_at(self, s):
        return [float(np.interp(s, self.s, self.points[:, 0])), float(np.interp(s, self.s, self.points[:, 1]))]


def intervals(mask):
    change = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1) - 1))


def clean_components(mask, min_area, seeds=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    if seeds is not None:
        seeded = np.unique(labels[seeds])
        keep &= np.isin(np.arange(count), seeded)
    return keep[labels]


def contours_of(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, .55, True).reshape(-1, 2).tolist() for c in contours if cv2.contourArea(c) >= 3]


class Analyzer:
    def __init__(self, reference_images, settings):
        self.settings = settings
        stack = np.stack(reference_images).astype(np.float32)
        self.background = np.median(stack, axis=0)
        self.wall = Wall(self.background, settings)
        self.noise = np.median(np.abs(stack - self.background), axis=0) * 1.4826
        h, w = self.background.shape
        self.roi = settings.roi or [int(w * .25), 3, int(w * .99), h - settings.edge_margin_px]
        x0, y0, x1, y1 = map(int, self.roi)
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('roi must be [left, top, right, bottom] within the image')
        self.allowed = np.zeros((h, w), bool)
        self.allowed[y0:y1, x0:x1] = True
        self.allowed &= ~self.wall.solid
        # A fixed strip on the solid side of the reference edge supports registration.
        # Liquid regions are excluded again by robust residual rejection below.
        self.reg_mask = ((self.wall.distance < -8) & (self.wall.distance > -65) &
                         (np.indices((h, w))[0] > self.wall.y_min + 20)).astype(np.uint8) * 255
        self.ref_grad = cv2.Sobel(self.background, cv2.CV_32F, 1, 0, ksize=3)

    def align(self, image):
        settings = self.settings
        flags = []
        warp = np.eye(2, 3, dtype=np.float32)
        if settings.register:
            grad = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            residual = np.abs(image.astype(np.float32) - self.background)
            mask = self.reg_mask.copy()
            mask[residual > 18] = 0
            try:
                score, warp = cv2.findTransformECC(self.ref_grad, grad, warp, cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4), mask, 5)
                if score < .7 or np.linalg.norm(warp[:, 2]) > 3:
                    flags.append('registration_uncertain')
                    warp = np.eye(2, 3, dtype=np.float32)
            except cv2.error:
                flags.append('registration_uncertain')
                warp = np.eye(2, 3, dtype=np.float32)
        h, w = image.shape
        aligned = cv2.warpAffine(image, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        # Remove small global illumination shifts using unchanged solid pixels.
        residual = aligned.astype(np.float32) - self.background
        static = (self.reg_mask > 0) & (np.abs(residual) < 12)
        illumination = float(np.median(residual[static])) if np.any(static) else 0.
        return np.clip(aligned.astype(np.float32) - illumination, 0, 255), warp[:, 2].tolist(), flags

    def segment(self, aligned):
        cfg, wall = self.settings, self.wall
        delta = self.background - aligned
        low = np.maximum(cfg.dark_support, self.noise * 4)
        high = np.maximum(cfg.dark_seed, self.noise * 7)
        support = (delta > low) & self.allowed
        seeds = (delta > high) & self.allowed
        support = clean_components(support, cfg.min_component_area, seeds)
        # Locate edges relative to each moving object's contrast, rather than
        # retaining every small background change. Low-contrast optical tails
        # can connect a drop to a blurred wall even when no liquid is there.
        contrast = self.contrast_field(support, delta)
        object_contrast = contrast
        edge_seeds = ((delta > np.maximum(high, cfg.contact_seed_fraction * contrast)) & self.allowed)
        # Near-wall thin-film components may have little contrast of their own.
        # Their contact still needs the independent temporal evidence below.
        edge_seeds |= seeds & (wall.distance <= 7)
        # A thin transparent film can have much less contrast than its thick
        # front bead. Use a local contrast envelope for edges, capped by the
        # object's robust contrast, so the bead does not erase the whole film.
        radius = cfg.edge_contrast_radius_px
        local_contrast = cv2.dilate(cv2.medianBlur(np.maximum(delta,0).astype(np.float32),3),
                                   np.ones((2*radius+1,2*radius+1),np.uint8))
        contrast = np.minimum(contrast, local_contrast)
        support &= delta > np.maximum(low, cfg.edge_contrast_fraction * contrast)
        support = clean_components(support, cfg.min_component_area, edge_seeds)
        radius = cfg.closing_radius
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            closed = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            # Never create a liquid-to-wall bridge in the contact band.
            support |= closed & (wall.distance > 7)
        joined = support | wall.solid
        # Filling a closed external silhouette includes refractive bright interiors.
        # The static wall closes attached regions without tracing internal highlights.
        filled = ndi.binary_fill_holes(joined)
        liquid = clean_components(filled & self.allowed, cfg.min_component_area)
        # A moving droplet can cast a narrow shadow onto the blurred wall even
        # before impact. Keep only components with an actual protruding body.
        count, labels = cv2.connectedComponents(liquid.astype(np.uint8), connectivity=8)
        supported_labels = np.unique(labels[liquid & (wall.distance > cfg.min_film_thickness_px)])
        liquid &= np.isin(labels, supported_labels[supported_labels != 0])
        repaired = np.zeros_like(liquid)
        if cfg.highlight_gap_radius:
            liquid, repaired = complete_bright_cavities(liquid,aligned,wall.distance,self.allowed,
                                                       cfg.highlight_gap_radius,cfg.cavity_boundary_fraction,
                                                       wall.solid,cfg.wall_uncertainty_px+.5)
        # Remove weak, unresolved fringes that run along the dry wall outside
        # the resolved liquid body. Enclosed bright interiors retain their
        # adjacent body support; independently strong thin film is retained.
        near_wall = wall.distance <= cfg.wall_uncertainty_px + .5
        body = liquid & ~near_wall
        if body.any():
            fringe = (liquid & near_wall & (ndi.distance_transform_edt(~body) > 2) &
                      (delta < np.maximum(cfg.dark_seed,cfg.contact_seed_fraction*object_contrast)))
            liquid &= ~fringe
            repaired &= liquid
        reconstructed = liquid & ~support
        return liquid, reconstructed, repaired, delta

    def contrast_field(self, mask, delta):
        """Robust contrast per connected object, measured outside wall blur.

        Enclosed bright interiors and near-wall shadow pixels do not set the
        reference contrast. The percentile tolerates isolated dark noise.
        """
        count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        values = np.zeros(count, np.float32)
        body = mask & (self.wall.distance > 7) & (delta > self.settings.dark_seed)
        for label in np.unique(labels[body]):
            samples = delta[body & (labels == label)]
            if len(samples) >= self.settings.min_component_area:
                values[label] = np.percentile(samples, 90)
        return values[labels]

    def contact_seed_profile(self, liquid, delta):
        """Independent photometric evidence at the reference wall itself.

        A geometric near-wall tolerance is not sufficient to initiate contact.
        Use a half-contrast seed; weaker attached intervals can subsequently be
        followed through time. Sampling an explicitly raster-aligned wall may
        otherwise land entirely inside a solid pixel, so move outward only as
        far as the local reference transition midpoint (at most one pixel).
        """
        wall, cfg = self.wall, self.settings
        contrast = wall.sample(self.contrast_field(liquid, delta), nearest=True).max(axis=1)
        profile = wall.sample(delta)
        ref = wall.sample(self.background)
        inside = wall.points - wall.normals * 7
        solid = cv2.remap(self.background, inside[:, 0:1], inside[:, 1:2], cv2.INTER_LINEAR).ravel()
        outside_index = min(int(round(7 / cfg.normal_step_px)), len(wall.offsets)-1)
        midpoint = (solid + ref[:, outside_index]) / 2
        boundary = (ref >= midpoint[:, None]) & (wall.offsets[None, :] <= 1)
        index = np.argmax(boundary, axis=1)
        at_wall = profile[np.arange(len(profile)), index]
        noise = wall.sample(self.noise)[np.arange(len(profile)), index]
        strong = (contrast > 0) & (at_wall >= np.maximum(cfg.contact_seed_fraction * contrast, noise * 7))
        # The same one-sample dropout tolerance used for contact geometry.
        # This only joins evidence samples; it does not fill image-space air gaps.
        strong = ndi.binary_closing(strong, structure=np.ones(3))
        return keep_runs(strong, cfg.min_contact_length_px)

    def measure(self, liquid, delta, contact_mask=None):
        cfg, wall = self.settings, self.wall
        # Temporal silhouette completion may infer an outer envelope but cannot
        # establish additional wall contact. Only the spatial mask supplies the
        # protected near-wall band; manually supplied geometry uses itself.
        contact_geometry=liquid
        if contact_mask is not None:
            contact_geometry=liquid.copy()
            band=wall.distance<=7
            contact_geometry[band]=contact_mask[band]
        sampled = wall.sample(contact_geometry, nearest=True) > .5
        sampled &= wall.sample_valid
        # Only a contiguous liquid run starting within the wall-location tolerance
        # can contribute contact. Thickness uses the attached outer envelope.
        has = sampled.any(axis=1)
        first = np.argmax(sampled, axis=1)
        distance_first = wall.offsets[first]
        possible = has & (distance_first <= cfg.wall_uncertainty_px + .5)
        step = cfg.normal_step_px
        columns = np.arange(sampled.shape[1])[None, :]
        after_first = columns >= first[:, None]
        stops = (~sampled) & after_first
        stop = np.argmax(stops, axis=1)
        stop[~stops.any(axis=1)] = sampled.shape[1]
        height = np.maximum(0, stop * step - step / 2)
        height[~possible] = 0
        # A transparent but enclosed interior is still liquid. Requiring a dark
        # near-wall pixel here would falsely split the contact at bright rims.
        confident = possible & (height >= cfg.min_film_thickness_px)
        confident = ndi.binary_closing(confident, structure=np.ones(3)) & possible
        # Suppress isolated one-pixel contacts; never bridge separate contact runs.
        possible = keep_runs(possible, cfg.min_contact_length_px)
        confident = keep_runs(confident, cfg.min_contact_length_px)
        segments = []
        possible_runs = intervals(possible)
        for start, end in intervals(confident):
            ss = max(0., wall.s[start] - .5)
            se = min(float(wall.s[-1]), wall.s[end] + .5)
            candidate = next((r for r in possible_runs if r[0] <= start <= r[1]), (start,end))
            pstart,pend = candidate
            segments.append({'s_start_px': float(ss), 's_end_px': float(se),
                'start': wall.point_at(ss), 'end': wall.point_at(se),
                'length_px': float(se-ss), 'confident_length_px': float(end-start+1),
                'start_range': [wall.point_at(float(wall.s[pstart])), wall.point_at(ss)],
                'end_range': [wall.point_at(se), wall.point_at(float(wall.s[pend]))]})
        height,maximum,trunc = self.outer_thickness(liquid,confident)
        return segments, maximum, height, possible, confident, trunc

    def outer_thickness(self, liquid, contact_selection):
        """Wall-normal envelope of components attached at supported contacts.

        Attachment is a two-dimensional connectivity property. A connected front
        rim remains eligible even when a particular normal crosses a bright
        cavity or an overhang. Independent airborne components remain excluded.
        Contact intervals are never enlarged by this envelope measurement.
        """
        cfg,wall = self.settings,self.wall
        _,labels = cv2.connectedComponents(liquid.astype(np.uint8),connectivity=8)
        sampled = wall.sample(labels,nearest=True).astype(np.int32)
        sampled[~wall.sample_valid] = 0
        occupied = sampled > 0
        first = np.argmax(occupied,axis=1)
        near = occupied.any(axis=1) & (wall.offsets[first] <= cfg.wall_uncertainty_px+.5) & contact_selection
        attached = np.unique(sampled[np.flatnonzero(near),first[near]])
        attached = attached[attached>0]
        selected = np.isin(sampled,attached) & occupied
        has = selected.any(axis=1)
        last = selected.shape[1]-1-np.argmax(selected[:,::-1],axis=1)
        # Intersect normals with exactly the polygons displayed in the viewer.
        # Pixel run endpoints otherwise overshoot the drawn contour by up to a
        # pixel and can select a quantization artefact as the maximum.
        attached_mask = liquid & np.isin(labels,attached)
        height = np.zeros(len(wall.s),np.float64)
        points,normals = wall.points.astype(np.float64),wall.normals.astype(np.float64)
        for contour in contours_of(attached_mask):
            vertices = np.asarray(contour,np.float64)
            vectors = np.roll(vertices,-1,axis=0)-vertices
            q = vertices[None,:,:]-points[:,None,:]
            den = normals[:,0,None]*vectors[None,:,1]-normals[:,1,None]*vectors[None,:,0]
            safe = np.abs(den)>1e-10
            t = np.divide(q[:,:,0]*vectors[None,:,1]-q[:,:,1]*vectors[None,:,0],den,
                          out=np.zeros_like(den),where=safe)
            u = np.divide(q[:,:,0]*normals[:,1,None]-q[:,:,1]*normals[:,0,None],den,
                          out=np.zeros_like(den),where=safe)
            valid = safe & (t>=0) & (u>=-1e-8) & (u<=1+1e-8) & has[:,None]
            height = np.maximum(height,np.where(valid,t,0).max(axis=1))
        trunc = bool(np.any(height>cfg.max_normal_distance_px) or np.any(has & (last==sampled.shape[1]-1)))
        height = np.minimum(height,cfg.max_normal_distance_px).astype(np.float32)
        maximum = None
        if np.any(height>0):
            k = int(np.argmax(height))
            origin = wall.points[k]
            end = origin + wall.normals[k] * height[k]
            start = int(np.argmax(selected[k]))
            maximum = {'value_px':float(height[k]),'s_px':float(wall.s[k]),
                       'start':origin.tolist(),'end':end.tolist(),'supported_contact':True,
                       'local_contact':bool(contact_selection[k]),
                       'spans_visible_gap':bool(wall.offsets[start]>cfg.wall_uncertainty_px+.5 or
                                                not selected[k,start:last[k]+1].all()),
                       'definition':'wall_normal_outer_envelope_of_attached_component'}
        return height,maximum,trunc

    def analyze(self, image):
        if image.shape != self.background.shape:
            raise ValueError('Frame dimensions changed within sequence')
        aligned, translation, flags = self.align(image)
        liquid, reconstructed, repaired, delta = self.segment(aligned)
        segments, maximum, height, possible, confident, trunc = self.measure(liquid, delta)
        yy, xx = np.nonzero(liquid)
        area = int(len(xx))
        bbox = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)] if area else None
        centroid = [float(xx.mean()), float(yy.mean())] if area else None
        minimum_gap = float(np.min(self.wall.distance[liquid])) if area else None
        if area:
            x0,y0,x1,y1 = self.roi
            if xx.min() <= x0 or yy.min() <= y0 or xx.max() >= x1-1 or yy.max() >= y1-1:
                flags.append('liquid_at_roi_boundary')
        uncertain_length = float(np.count_nonzero(possible & ~confident))
        if uncertain_length > 8:
            flags.append('contact_boundary_uncertain')
        if trunc:
            flags.append('normal_search_truncated')
        if maximum and not maximum['supported_contact']:
            flags.append('thickness_contact_uncertain')
        if np.count_nonzero(repaired) > 20:
            flags.append('highlight_boundary_reconstructed')
        data = {'area_px2': area, 'centroid': centroid, 'bbox': bbox,
                'contours': contours_of(liquid), 'contact_segments': segments,
                'contact_length_px': sum(s['length_px'] for s in segments),
                'confident_contact_length_px': float(np.count_nonzero(confident)),
                'uncertain_contact_length_px': uncertain_length,
                'possible_contact_length_px': float(np.count_nonzero(possible)),
                'maximum_visible_thickness': maximum,
                'minimum_wall_gap_px': minimum_gap,
                'registration_translation_px': translation,
                'filled_interior_area_px2': int(np.count_nonzero(reconstructed)),
                'repaired_boundary_area_px2': int(np.count_nonzero(repaired)),
                'flags': flags, 'manual_reviewed': False}
        return data, liquid, height, possible, confident, aligned.astype(np.uint8)


def keep_runs(mask, min_length):
    output = np.zeros_like(mask)
    for start,end in intervals(mask):
        if end-start+1 >= min_length:
            output[start:end+1] = True
    return output
