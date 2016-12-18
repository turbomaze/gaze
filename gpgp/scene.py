"""
GPGP
@author Anthony Liu <igliu@mit.edu>
@version 0.2.1
"""

from context import Rasta
import numpy as np


class Scene(object):
    @classmethod
    def sample(cls, num_boxes):
        # hardcoded range parameters
        x_range, y_range, z_range = 13.33333, 10., 10.

        # generate the boxes
        boxes = []
        for i in range(num_boxes):
            # generate random box
            boxes.append([
                np.random.uniform(0, x_range/2. - 1.),
                np.random.uniform(0, y_range - 1.),
                np.random.uniform(0, z_range - 1.)
            ])

        # get the viewer
        viewer_center = [0.75*x_range, 0, z_range/2.]
        target_box = np.random.choice(num_boxes)
        viewer = [
            np.random.normal(viewer_center[0], x_range/16.),
            boxes[target_box][1],
            np.random.normal(viewer_center[2], z_range/2.),
            target_box
        ]
        viewer[0] = min(x_range - 1., max(0, viewer[0]))
        viewer[2] = min(z_range - 1., max(0, viewer[2]))

        return {
            'viewer': viewer,
            'boxes': boxes
        }

    @classmethod
    def get_model_from_scene(cls, scene):
        boxes = []

        v1 = scene['boxes'][scene['viewer'][3]]
        v2 = np.subtract(scene['viewer'][0:3], v1)
        v3 = [1, 0, 0]
        angle = np.arccos(np.dot(v2, v3) / (
            np.linalg.norm(v2) * np.linalg.norm(v3)
        ))
        viewer_model = Rasta.rotate_box(
            Rasta.get_box([
                scene['viewer'][0],
                scene['viewer'][1],
                scene['viewer'][2]
            ], 1, 1, 1, color=(255, 0, 0)),
            angle
        )
        viewer_model['faces'][0][1] = (0, 1e6, 0)
        # broken for some reason
        viewer_model['faces'][1][0] = list(reversed(
            viewer_model['faces'][1][0]
        ))
        viewer_model['faces'][1][1] = (0, 1e6, 1e6)
        boxes.append(viewer_model)

        for box in scene['boxes']:
            boxes.append(Rasta.get_box(
                box,
                1, 1, 1,
                color=(0, 0, 255)
            ))

        return boxes
