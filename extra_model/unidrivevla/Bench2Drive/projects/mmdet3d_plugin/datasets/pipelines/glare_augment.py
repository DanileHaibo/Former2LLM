import numpy as np
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class LeadVehicleGlare(object):
    """Apply a bright Gaussian glare over the nearest front vehicle in CAM_FRONT.

    The transform uses GT boxes only to choose and project the lead vehicle. It is
    intended for controlled open-loop attack evaluation, not for training.
    """

    def __init__(
        self,
        cam_index=0,
        vehicle_labels=(0, 1, 2),
        min_forward_y=2.0,
        max_forward_y=80.0,
        max_lateral_x=14.0,
        margin_px=18,
        sigma_scale=0.55,
        peak_bgr=(255.0, 255.0, 255.0),
        min_area_frac=0.0002,
        max_area_frac=0.38,
        min_depth=0.5,
        solid_blend=0.35,
    ):
        self.cam_index = cam_index
        self.vehicle_labels = set(vehicle_labels)
        self.min_forward_y = min_forward_y
        self.max_forward_y = max_forward_y
        self.max_lateral_x = max_lateral_x
        self.margin_px = int(margin_px)
        self.sigma_scale = float(sigma_scale)
        self.peak_bgr = np.asarray(peak_bgr, dtype=np.float32)
        self.min_area_frac = float(min_area_frac)
        self.max_area_frac = float(max_area_frac)
        self.min_depth = float(min_depth)
        self.solid_blend = float(solid_blend)

    def __call__(self, results):
        imgs = results.get("img")
        boxes = results.get("gt_bboxes_3d")
        labels = results.get("gt_labels_3d")
        lidar2img = results.get("lidar2img")

        meta = dict(applied=False, reason="missing_inputs")
        if imgs is None or boxes is None or labels is None or lidar2img is None:
            results["lead_vehicle_glare"] = meta
            return results
        if self.cam_index >= len(imgs) or self.cam_index >= len(lidar2img):
            meta["reason"] = "bad_cam_index"
            results["lead_vehicle_glare"] = meta
            return results

        idx = self._pick_lead_vehicle(np.asarray(boxes), np.asarray(labels))
        if idx is None:
            meta["reason"] = "no_front_vehicle"
            results["lead_vehicle_glare"] = meta
            return results

        img = np.asarray(imgs[self.cam_index]).astype(np.float32, copy=True)
        roi = self._project_roi(np.asarray(boxes)[idx], np.asarray(lidar2img[self.cam_index]), img.shape)
        if roi is None:
            meta.update(reason="projection_failed", lead_index=int(idx))
            results["lead_vehicle_glare"] = meta
            return results

        x1, y1, x2, y2 = roi
        area_frac = ((x2 - x1) * (y2 - y1)) / float(img.shape[0] * img.shape[1])
        if area_frac < self.min_area_frac or area_frac > self.max_area_frac:
            meta.update(reason="roi_area_out_of_range", lead_index=int(idx), roi=roi, area_frac=area_frac)
            results["lead_vehicle_glare"] = meta
            return results

        img = self._apply_gaussian_glare(img, roi)
        imgs[self.cam_index] = img.astype(np.float32)
        meta.update(applied=True, reason="ok", lead_index=int(idx), roi=roi, area_frac=area_frac)
        results["lead_vehicle_glare"] = meta
        return results

    def _pick_lead_vehicle(self, boxes, labels):
        if boxes.size == 0:
            return None
        best_idx = None
        best_forward = None
        for i, box in enumerate(boxes):
            if int(labels[i]) not in self.vehicle_labels:
                continue
            lateral_x = float(box[0])
            forward_y = float(box[1])
            if not (self.min_forward_y <= forward_y <= self.max_forward_y):
                continue
            if abs(lateral_x) > self.max_lateral_x:
                continue
            if best_forward is None or forward_y < best_forward:
                best_idx = i
                best_forward = forward_y
        return best_idx

    def _project_roi(self, box, lidar2img, img_shape):
        corners = self._box_corners(box)
        hom = np.concatenate([corners, np.ones((corners.shape[0], 1), dtype=np.float32)], axis=1)
        proj = hom @ lidar2img.T
        valid = proj[:, 2] > self.min_depth
        if valid.sum() < 4:
            return None
        uv = proj[valid, :2] / np.maximum(proj[valid, 2:3], 1e-6)
        h, w = img_shape[:2]
        x1, y1 = np.percentile(uv, [3], axis=0)[0]
        x2, y2 = np.percentile(uv, [97], axis=0)[0]
        x1 = int(np.floor(max(0, x1 - self.margin_px)))
        y1 = int(np.floor(max(0, y1 - self.margin_px)))
        x2 = int(np.ceil(min(w - 1, x2 + self.margin_px)))
        y2 = int(np.ceil(min(h - 1, y2 + self.margin_px)))
        if x2 <= x1 + 2 or y2 <= y1 + 2:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _box_corners(box):
        x, y, z = box[:3].astype(np.float32)
        dx, dy, dz = np.maximum(box[3:6].astype(np.float32), 0.1)
        yaw = float(box[6])
        local = np.array(
            [
                [dx / 2, dy / 2, dz / 2],
                [dx / 2, -dy / 2, dz / 2],
                [-dx / 2, -dy / 2, dz / 2],
                [-dx / 2, dy / 2, dz / 2],
                [dx / 2, dy / 2, -dz / 2],
                [dx / 2, -dy / 2, -dz / 2],
                [-dx / 2, -dy / 2, -dz / 2],
                [-dx / 2, dy / 2, -dz / 2],
            ],
            dtype=np.float32,
        )
        rot = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return local @ rot.T + np.array([x, y, z], dtype=np.float32)

    def _apply_gaussian_glare(self, img, roi):
        x1, y1, x2, y2 = roi
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        sx = max((x2 - x1) * self.sigma_scale, 8.0)
        sy = max((y2 - y1) * self.sigma_scale, 8.0)
        glare = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
        roi_mask = np.zeros((h, w), dtype=np.float32)
        roi_mask[y1:y2, x1:x2] = 1.0
        glare = np.maximum(glare, roi_mask * self.solid_blend)[..., None]
        return np.clip(img + glare * self.peak_bgr[None, None, :], 0, 255)

