import numpy as np
import cv2

def get_default_anchors():
    return {
        "steps": [8, 16, 32],
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
    }


class SCRFDPostProc(object):
    # The following params are corresponding to those used for training the model
    NUM_CLASSES = 1
    NUM_LANDMARKS = 10
    LABEL_OFFSET = 1

    def __init__(self, image_dims=(300, 300), nms_iou_thresh=0.6, score_threshold=0.3, anchors=None):
        if not anchors:
            anchors = get_default_anchors()
        self._image_dims = image_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_branches = len(anchors["steps"])
        self._anchors = self.extract_anchors(anchors["min_sizes"], anchors["steps"])

    def collect_box_class_predictions(self, output_branches):
        box_predictors_list = []
        class_predictors_list = []
        landmarks_predictors_list = []
        sorted_output_branches = output_branches
        num_branches = self._num_branches
        assert len(sorted_output_branches) % num_branches == 0, "All branches must have the same number of output nodes"
        num_output_nodes_per_branch = len(sorted_output_branches) // num_branches
        for branch_index in range(0, len(sorted_output_branches), num_output_nodes_per_branch):
            box_predictors_list.append(np.reshape(sorted_output_branches[branch_index], (-1, 4)))
            class_predictors_list.append(
                np.reshape(sorted_output_branches[branch_index + 1], (-1, self.NUM_CLASSES))
            )

            if num_output_nodes_per_branch > 2:
                # Assume output is landmarks
                landmarks_predictors_list.append(
                    np.reshape(sorted_output_branches[branch_index + 2], (-1, 10))
                )
        box_predictors = np.concatenate(box_predictors_list, axis=0)
        class_predictors = np.concatenate(class_predictors_list, axis=0)
        landmarks_predictors = np.concatenate(landmarks_predictors_list, axis=0) if landmarks_predictors_list else None
        return box_predictors, class_predictors, landmarks_predictors

    def extract_anchors(self, min_sizes, steps):
        anchors = []
        for stride, min_size in zip(steps, min_sizes, strict=True):
            height = self._image_dims[0] // stride
            width = self._image_dims[1] // stride
            num_anchors = len(min_size)

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers[:, 0] /= self._image_dims[0]
            anchor_centers[:, 1] /= self._image_dims[1]
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            anchor_scales = np.ones_like(anchor_centers, dtype=np.float32) * stride
            anchor_scales[:, 0] /= self._image_dims[0]
            anchor_scales[:, 1] /= self._image_dims[1]
            anchor = np.concatenate([anchor_centers, anchor_scales], axis=1)
            anchors.append(anchor)
        return np.concatenate(anchors, axis=0)

    def _decode_landmarks(self, landmarks_detections, anchors):
        preds = []
        for i in range(0, self.NUM_LANDMARKS, 2):
            px = anchors[:, 0] + landmarks_detections[:, i] * anchors[:, 2]
            py = anchors[:, 1] + landmarks_detections[:, i + 1] * anchors[:, 3]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def _decode_boxes(self, box_detections, anchors):
        x1 = anchors[:, 0] - box_detections[:, 0] * anchors[:, 2]
        y1 = anchors[:, 1] - box_detections[:, 1] * anchors[:, 3]
        x2 = anchors[:, 0] + box_detections[:, 2] * anchors[:, 2]
        y2 = anchors[:, 1] + box_detections[:, 3] * anchors[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def postprocess(self, endnodes):
        box_predictions, classes_predictions, landmarks_predictors = self.collect_box_class_predictions(endnodes)
        scores = classes_predictions[:, 0]
        boxes = self._decode_boxes(box_predictions, self._anchors)
        landmarks = None
        if landmarks_predictors is not None:
            landmarks = self._decode_landmarks(landmarks_predictors, self._anchors)

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self._score_threshold,
            self._nms_iou_thresh
        )
        boxes = boxes[indices]
        scores = scores[indices]
        if landmarks is not None:
            landmarks = landmarks[indices]

        results = {
            "detection_boxes": boxes,
            "detection_scores": scores,
            "detection_classes": np.ones(len(scores), dtype=np.int16),
            "num_detections": len(scores),
        }

        if landmarks is not None:
            results["face_landmarks"] = landmarks

        return results