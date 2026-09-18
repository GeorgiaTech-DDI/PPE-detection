import numpy as np
import tensorflow as tf



class SCRFDPostProc(object):
    # The following params are corresponding to those used for training the model
    NUM_CLASSES = 1
    NUM_LANDMARKS = 10
    LABEL_OFFSET = 1

    def __init__(self, image_dims=(300, 300), nms_iou_thresh=0.6, score_threshold=0.3, anchors=None):
        self._image_dims = image_dims
        self._nms_iou_thresh = nms_iou_thresh
        self._score_threshold = score_threshold
        self._num_branches = len(anchors["steps"])
        if anchors is None:
            raise ValueError("Missing detection anchors metadata")
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
            num_of_batches, _, _, _ = tf.unstack(tf.shape(sorted_output_branches[branch_index]))
            box_predictors_list.append(tf.reshape(sorted_output_branches[branch_index], shape=[num_of_batches, -1, 4]))
            class_predictors_list.append(
                tf.reshape(sorted_output_branches[branch_index + 1], shape=[num_of_batches, -1, self.NUM_CLASSES])
            )

            if num_output_nodes_per_branch > 2:
                # Assume output is landmarks
                landmarks_predictors_list.append(
                    tf.reshape(sorted_output_branches[branch_index + 2], shape=[num_of_batches, -1, 10])
                )
        box_predictors = tf.concat(box_predictors_list, axis=1)
        class_predictors = tf.concat(class_predictors_list, axis=1)
        landmarks_predictors = tf.concat(landmarks_predictors_list, axis=1) if landmarks_predictors_list else None
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
        return tf.convert_to_tensor(np.concatenate(anchors, axis=0))

    def _decode_landmarks(self, landmarks_detections, anchors):
        preds = []
        for i in range(0, self.NUM_LANDMARKS, 2):
            px = anchors[:, 0] + landmarks_detections[:, i] * anchors[:, 2]
            py = anchors[:, 1] + landmarks_detections[:, i + 1] * anchors[:, 3]
            preds.append(px)
            preds.append(py)
        return tf.stack(preds, axis=-1)

    def _decode_boxes(self, box_detections, anchors):
        x1 = anchors[:, 0] - box_detections[:, 0] * anchors[:, 2]
        y1 = anchors[:, 1] - box_detections[:, 1] * anchors[:, 3]
        x2 = anchors[:, 0] + box_detections[:, 2] * anchors[:, 2]
        y2 = anchors[:, 1] + box_detections[:, 3] * anchors[:, 3]
        return tf.stack([x1, y1, x2, y2], axis=-1)

    def tf_postproc(self, endnodes):
        with tf.name_scope("Postprocessor"):
            box_predictions, classes_predictions, landmarks_predictors = self.collect_box_class_predictions(endnodes)
            additional_fields = {}

            detection_scores = classes_predictions

            batch_size, num_proposals = tf.unstack(tf.slice(tf.shape(box_predictions), [0], [2]))

            tiled_anchor_boxes = tf.tile(tf.expand_dims(self._anchors, 0), [batch_size, 1, 1])
            tiled_anchors_boxlist = tf.reshape(tiled_anchor_boxes, [-1, 4])

            decoded_boxes = self._decode_boxes(tf.reshape(box_predictions, (-1, 4)), tiled_anchors_boxlist)
            detection_boxes = tf.reshape(decoded_boxes, [batch_size, num_proposals, 4])

            decoded_landmarks = None
            if tf.is_tensor(landmarks_predictors):
                decoded_landmarks = self._decode_landmarks(
                    tf.reshape(landmarks_predictors, (-1, 10)), tiled_anchors_boxlist
                )
                decoded_landmarks = tf.reshape(decoded_landmarks, [batch_size, num_proposals, 10])
                additional_fields["landmarks"] = decoded_landmarks

            detection_boxes = tf.identity(tf.expand_dims(detection_boxes, axis=[2]), "raw_box_locations")
            scores = classes_predictions[0, :, 0].numpy()
            boxes = detection_boxes[0].numpy()
            landmarks = decoded_landmarks[0].numpy() if decoded_landmarks is not None else None

            keep = scores >= self._score_threshold

            boxes = boxes[keep]
            scores = scores[keep]

            if landmarks is not None:
                landmarks = landmarks[keep]

            # cv2 expects [x, y, width, height]
            xywh = boxes.copy()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]

            indices = cv2.dnn.NMSBoxes(
                xywh.tolist(),
                scores.tolist(),
                self._score_threshold,
                self._nms_iou_thresh
            )

            indices = np.asarray(indices).flatten()

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

        nmsed_additional_fields = nmsed_additional_fields or {}
        face_landmarks = nmsed_additional_fields.get("landmarks")
        if tf.is_tensor(face_landmarks):
            results["face_landmarks"] = face_landmarks

        return results