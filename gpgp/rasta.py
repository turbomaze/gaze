"""
GPGP
@author Anthony Liu <igliu@mit.edu>
@version 0.2.1
"""

import sys
import math
import numpy as np
from PIL import Image, ImageDraw


class Rasta(object):
    def __init__(
        self,
        camera_file,
        global_offset, global_scale,
        screen_width, screen_height
    ):
        self.camera = self.parse_camera_file(camera_file)
        self.global_offset = global_offset
        self.global_scale = global_scale
        self.screen_width = screen_width
        self.screen_height = screen_height

    def render(self, models, out_file):
        # prep the canvas
        data = np.zeros(
            (self.screen_height, self.screen_width, 3),
            dtype=np.uint8
        )
        data.fill(255)
        img = Image.fromarray(data, 'RGB')
        draw = ImageDraw.Draw(img, 'RGB')

        # merge the models
        model = self.merge_models(models)

        # sort the faces of the model
        sorted_faces = sorted(
            model['faces'],
            key=lambda face: -self.get_center(face)[0][2]
        )

        # add the boundary
        boundary = self.get_boundary_box()
        sorted_faces = boundary['faces'] + sorted_faces

        # iterate through the faces and render them
        for face, color, outline in sorted_faces:
            # apply the global offset
            face = [[
                self.global_scale[0] * point[0] +
                self.global_offset[0],
                self.global_scale[1] * point[1] +
                self.global_offset[1],
                self.global_scale[2] * point[2] +
                self.global_offset[2]
            ] for point in face]

            # get the screen coordinates of the vertices
            points = filter(
                lambda x: x is not None,
                map(lambda point: self.point_to_pixel(
                    point
                ), face)
            )
            if len(points) < 3:
                continue

            # get the face center and normal
            center = self.get_center(face)
            normal = self.get_normal(face)

            # always draw outlines
            if outline:
                draw.polygon(points, outline=color)

            # draw filled polygons if they're facing
            elif self.is_facing(
                center, self.camera['eye'], normal
            ):
                lit_color = self.get_lit_color(color, normal)
                if not outline:
                    draw.polygon(points, fill=lit_color)

        del draw
        img.save(out_file, 'PNG')

    def point_to_pixel(self, raw_p):
        p = np.array(raw_p + [1])
        p[2] /= 6.
        p_cam = np.dot(p, self.camera['cam_mat'])

        if -p_cam[2] < self.camera['near']:
            return None

        p_proj = np.dot(p_cam, self.camera['proj_mat'])
        p_proj /= p_proj[3]
        x = (p_proj[0] + 1.)/2. * self.screen_width
        y = (p_proj[1] + 1.)/2. * self.screen_height

        if x > self.screen_width or x < 0:
            return None
        if y > self.screen_height or y < 0:
            return None

        return (x, y)

    @classmethod
    def get_lit_color(cls, color, normal):
        light_1 = np.array([1, 2, 2])
        lightness_1 = abs(np.dot(normal, light_1))
        lightness_1 /= np.linalg.norm(light_1)
        light_2 = np.array([-1, 1, 2])
        lightness_2 = abs(np.dot(normal, light_2))
        lightness_2 /= np.linalg.norm(light_2)
        lightness = (lightness_1 + lightness_2) / 1.8
        return tuple(int(c * lightness) for c in color)

    @classmethod
    def get_normal(cls, face):
        v1 = np.subtract(face[1], face[0])
        v2 = np.subtract(face[2], face[1])
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)

    @classmethod
    def get_center(cls, face):
        sum = face[0]
        for i in range(1, len(face)):
            sum = np.add(sum, face[i])
        center = sum / float(len(face))
        return center

    @classmethod
    def is_facing(cls, center, eye, normal):
        return np.dot(np.subtract(-center, eye), normal) > 0

    @classmethod
    def get_camera(cls, pos, z_axis):
        pos = [-a for a in pos]
        z_axis = z_axis / np.linalg.norm(np.array(z_axis))
        z_axis = np.append(z_axis, 0)

        # get y rotation
        y_rot = math.atan2(z_axis[0], z_axis[2])
        y_rot_matrix = np.array([
            [math.cos(y_rot), 0, math.sin(y_rot), 0],
            [0, 1, 0, 0],
            [-math.sin(y_rot), 0, math.cos(y_rot), 0],
            [0, 0, 0, 1]
        ])

        # apply the y rotation to reduce the z axis
        z_axis_ = np.dot(y_rot_matrix, z_axis)

        # compute the x rotation
        x_rot = math.atan2(z_axis_[1], z_axis_[2])
        x_rot_matrix = np.array([
            [1, 0, 0, 0],
            [0, math.cos(x_rot), math.sin(x_rot), 0],
            [0, -math.sin(x_rot), math.cos(x_rot), 0],
            [0, 0, 0, 1]
        ])
        x_axis = np.dot(y_rot_matrix, np.array([1, 0, 0, 0]))
        x_axis = np.dot(x_rot_matrix, x_axis)
        y_axis = np.dot(y_rot_matrix, np.array([0, 1, 0, 0]))
        y_axis = np.dot(x_rot_matrix, y_axis)
        cam_mat = np.array([
            x_axis,
            y_axis,
            np.array(z_axis),
            np.array(pos + [1])
        ])
        return np.linalg.inv(cam_mat)

    @classmethod
    def parse_camera_file(cls, file_name):
        with open(file_name) as file:
            lines = filter(
                lambda x: not x.startswith('#') and len(x) != 0,
                [l.rstrip('\n') for l in file.readlines()]
            )

            if len(lines) < 3:
                sys.exit('Bad camera file %s' % file_name)

            camera_info = [float(x) for x in lines[0].split(',')]
            pos = cls.parse_point_list(lines[1])[0]
            z_axis = cls.parse_point_list(lines[2])[0]
            if camera_info is None:
                sys.exit('Bad camera file %s' % file_name)
            if pos is None or z_axis is None:
                sys.exit('Bad camera file %s' % file_name)

            # get the projection matrix
            near = camera_info[0]
            far = camera_info[1]
            aspect = camera_info[2]
            fovy = camera_info[3]
            cam_mat = cls.get_camera(pos, z_axis)
            proj_mat = np.array([
                [1./(aspect*np.tan(fovy/2.)), 0, 0, 0],
                [0, 1./np.tan(fovy/2.), 0, 0],
                [0, 0, -far/(far-near), -(near*far)/(far-near)],
                [0, 0, -1, 0],
            ])

            return {
                'near': near,
                'far': far,
                'aspect': aspect,
                'fovy': fovy,
                'eye': pos,
                'at': z_axis,
                'cam_mat': cam_mat,
                'proj_mat': proj_mat
            }
        sys.exit('Bad camera file %s' % file_name)

    @classmethod
    def parse_point_list(cls, points):
        try:
            return [
                [float(n), float(m), float(p)] for n, m, p in [
                    s.split(',') for s in points.split('|')
                ]
            ]
        except:
            return None

    @classmethod
    def rotate_box(cls, box, angle):
        box_ = {'faces': []}

        # construct the rotation matrix about the y-axis
        rot_mat = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        # compute the center of the box for shift purposes
        center = [0, 0, 0]
        for face in box['faces']:
            for point in face[0]:
                center = np.add(center, point)
        center = center/24.

        # shift, rotate, and unshift all the points
        for face in box['faces']:
            points = []
            for point in face[0]:
                point_ = np.dot(
                    rot_mat,
                    np.subtract(point, center)
                )
                points.append(np.add(point_, center))
            face_ = [points, face[1], face[2]]
            box_['faces'].append(face_)

        return box_

    @classmethod
    def get_box(cls, pos, l, w, h, color=False, outline=False):
        faces = []
        points = [
            [i >> 2, (i >> 1) & 1, i & 1] for i in range(2**3)
        ]
        points = [
            [l*a + pos[0], w*b + pos[1], h*c + pos[2]]
            for a, b, c in points
        ]
        faces_by_points = [
            [0, 1, 3, 2],  # x-axis
            [6, 7, 5, 4],  # x-axis
            [4, 5, 1, 0],  # y-axis
            [2, 3, 7, 6],  # y-axis
            [0, 2, 6, 4],  # z-axis
            [5, 7, 3, 1]  # z-axis
        ]

        for face_by_points in faces_by_points:
            if color is False:
                color = tuple([
                    max(min(
                        int(np.random.uniform() * 256),
                        255
                    ), 0) for _ in range(3)
                ])

            faces.append([
                [points[x] for x in face_by_points],
                color,
                outline
            ])

        return {'faces': faces}

    @classmethod
    def get_boundary_box(cls):
        boundary = Rasta.get_box(
            [0, 0, 0],
            13.3333, 10, 10,
            color=(0, 0, 0),
            outline=True
        )
        return boundary

    @classmethod
    def merge_models(cls, models):
        model = {'faces': []}
        for submodel in models:
            model['faces'] += submodel['faces']
        return model
