"""
GPGP
@author Anthony Liu <igliu@mit.edu>
@version 0.2.1
"""

from context import Rasta
import numpy as np


class Scene(object):
    x_range, y_range, z_range = 13.33333, 10., 10.
    num_viewer_params = 3
    num_box_params = 3

    @classmethod
    def sample(cls, num_boxes):
        # generate the boxes
        boxes = []
        for i in range(num_boxes):
            # generate random box
            boxes += [
                np.random.uniform(0, cls.x_range/2. - 1.),
                np.random.uniform(0, cls.y_range - 1.),
                np.random.uniform(0, cls.z_range - 1.)
            ]

        # get the viewer
        viewer_center = [0.75*cls.x_range, cls.z_range/2.]
        target_box = np.random.choice(num_boxes)
        viewer = [
            np.random.normal(
                viewer_center[0], cls.x_range/16.
            ),
            np.random.normal(
                viewer_center[1], cls.z_range/4.
            ),
            target_box
        ]
        viewer[0] = min(cls.x_range - 1., max(0, viewer[0]))
        viewer[1] = min(cls.z_range - 1., max(0, viewer[2]))

        return viewer + boxes

    @classmethod
    def transition(cls, latent, k):
        latent_ = latent[:]
        num_boxes = len(latent) - cls.num_viewer_params
        num_boxes /= cls.num_box_params

        # first few latent variables are the viewer
        if k == 0:
            latent_[k] = np.random.normal(latent[k], 2.)
            latent_[k] = min(
                cls.x_range - 1., max(0, latent_[k])
            )
        elif k == 1:
            latent_[k] = np.random.normal(latent[k], 2.)
            latent_[k] = min(
                cls.z_range - 1., max(0, latent_[k])
            )
        elif k == 2:
            latent_[k] = np.random.choice(num_boxes)

        # the rest are the boxes
        elif (k - cls.num_viewer_params) % 3 == 0:
            latent_[k] = np.random.normal(latent[k], 2.)
            latent_[k] = min(
                cls.x_range/2. - 1., max(0, latent_[k])
            )
        elif (k - cls.num_viewer_params) % 3 == 1:
            latent_[k] = np.random.normal(latent[k], 2.)
            latent_[k] = min(
                cls.y_range - 1., max(0, latent_[k])
            )
        elif (k - cls.num_viewer_params) % 3 == 2:
            latent_[k] = np.random.normal(latent[k], 2.)
            latent_[k] = min(
                cls.z_range - 1., max(0, latent_[k])
            )
        else:
            return latent

        return latent_

    @classmethod
    def get_model_from_latent(cls, latent):
        viewer_offset = cls.num_viewer_params
        num_boxes = len(latent) - cls.num_viewer_params
        num_boxes /= cls.num_box_params

        latent[2] = min(num_boxes - 1, max(0, int(latent[2])))
        scene = {
            'viewer': [
                latent[0],
                latent[
                    cls.num_viewer_params +
                    latent[2]*cls.num_box_params + 1
                ],
                latent[1],
                latent[2]
            ],
            # go from a list to a list of lists
            'boxes': [
                latent[
                    viewer_offset +
                    cls.num_box_params * i:viewer_offset +
                    cls.num_box_params * i + cls.num_box_params
                ]
                for i in range(num_boxes)
            ]
        }

        boxes = []

        v1 = scene['boxes'][scene['viewer'][3]]
        v2 = np.subtract(
            scene['viewer'][0:3], v1
        )
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
