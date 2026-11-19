import numpy as np
import torch
from ultralytics.utils.tal import dist2bbox, make_anchors


class Yolo26Decoder:
    """Decodes raw YOLO26 head outputs into per-class detection arrays.

    Input:  {output_name: (1, H, W, C) float32}, the raw HEF outputs.
    Output: (1, num_classes, max_det, 5) float32, rows of
            [ymin, xmin, ymax, xmax, score], normalized to 0-1.
            Classes with fewer detections are zero-padded to the same length.

    Create one instance per HEF and use it from one thread, since it caches
    the anchor grid and the resolved output order.
    """

    def __init__(self, output_names, input_hw, conf_thr=0.25, max_det=300):
        self.input_h, self.input_w = input_hw
        self.conf_thr = conf_thr
        self.max_det = max_det
        self._names_in = list(output_names)
        self._names = None       # box branches first, then class branches
        self._anchors = None
        self.num_classes = None

    def __call__(self, outs: dict) -> np.ndarray:
        with torch.no_grad():
            dets = self._decode(outs)               # (300, 6) pixels xyxy
        dets = dets[dets[:, 4] > self.conf_thr]
        return self._to_per_class_layout(dets)

    # -- decoding ----------------------------------------------------------

    def _resolve_order(self, outs):
        # Box branches have 4 channels (no DFL in YOLO26), class branches have nc.
        # Sort each group by map height, largest first (stride 8 -> 32).
        by_size = lambda n: -outs[n].shape[1]
        box = sorted([n for n in self._names_in if outs[n].shape[-1] == 4], key=by_size)
        cls = sorted([n for n in self._names_in if outs[n].shape[-1] != 4], key=by_size)
        assert box and len(box) == len(cls), "unexpected output layout (is nc == 4?)"
        for b, c in zip(box, cls):
            assert outs[b].shape[1:3] == outs[c].shape[1:3], "box/class scale mismatch"
        self._names = box + cls
        self.num_classes = outs[cls[0]].shape[-1]

    def _decode(self, outs) -> np.ndarray:
        if self._names is None:
            self._resolve_order(outs)
        arrs = [outs[n] for n in self._names]
        split = len(arrs) // 2
        box_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in arrs[:split]]
        cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in arrs[split:]]

        if self._anchors is None:
            strides = [self.input_h / x.shape[2] for x in box_maps]
            self._anchors = make_anchors(box_maps, strides)
        anchors, stride = self._anchors

        boxes = torch.cat([x.flatten(2) for x in box_maps], 2).transpose(1, 2)
        boxes = dist2bbox(boxes, anchors, xywh=False) * stride
        scores = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2).sigmoid()

        C = scores.shape[2]
        k = min(self.max_det, scores.shape[1])
        idx = scores.amax(-1).topk(k, dim=1).indices[..., None]
        boxes = boxes.gather(1, idx.expand(-1, -1, 4))
        scores = scores.gather(1, idx.expand(-1, -1, C))

        s, i = scores.flatten(1).topk(min(self.max_det, k * C), dim=1)
        boxes = boxes.gather(1, (i // C)[..., None].expand(-1, -1, 4))
        out = torch.cat((boxes, s[..., None], (i % C)[..., None].float()), 2)
        return out[0].numpy()                        # batch size is 1

    # -- layout conversion -------------------------------------------------

    def _to_per_class_layout(self, dets: np.ndarray) -> np.ndarray:
        """(N, 6) pixel xyxy rows -> (1, num_classes, max_det, 5) normalized yxyx + score."""
        per_class = []
        for c in range(self.num_classes):
            d = dets[dets[:, 5] == c]
            per_class.append(
                np.stack([d[:, 1] / self.input_h, d[:, 0] / self.input_w,
                          d[:, 3] / self.input_h, d[:, 2] / self.input_w,
                          d[:, 4]], axis=1).astype(np.float32)
                if len(d) else np.zeros((0, 5), np.float32)
            )
        longest = max(len(a) for a in per_class)
        padded = np.stack([
            np.pad(a, ((0, longest - len(a)), (0, 0))) for a in per_class
        ])
        return padded[None]                          # (1, C, longest, 5)