import sys
import math
import numpy as np
from PIL import Image, ImageDraw


def render(camera, model):
    # prep the canvas
    data = np.zeros((600, 800, 3), dtype=np.uint8)
    data.fill(255)
    img = Image.fromarray(data, 'RGB')
    draw = ImageDraw.Draw(img, 'RGB')

    for face, color, outline in model['faces']:
        # get the screen coordinates of the vertices
        points = filter(
            lambda x: x is not None,
            map(lambda f: point_to_pixel(camera, f), face)
        )
        if len(points) < 3:
            continue

        # get the face center and normal
        center = get_center(face)
        normal = get_normal(face)

        if is_facing(center, camera['eye'], normal):
            lit_color = get_lit_color(color, normal)

            # draw the polygon
            if outline:
                draw.polygon(points, outline=lit_color)
            else:
                draw.polygon(points, fill=lit_color)

    del draw
    img.save('out.png', 'PNG')

def get_lit_color(color, normal):
    light_1 = np.array([1, 2, 2])
    lightness_1 = abs(np.dot(normal, light_1))
    lightness_1 /= np.linalg.norm(light_1)
    light_2 = np.array([-1, 1, 2])
    lightness_2 = abs(np.dot(normal, light_2))
    lightness_2 /= np.linalg.norm(light_2)
    lightness = (lightness_1 + lightness_2) / 1.8
    return tuple(int(c * lightness) for c in color)


def get_normal(face):
    v1 = np.subtract(face[1], face[0])
    v2 = np.subtract(face[2], face[1])
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)


def get_center(face):
    sum = face[0]
    for i in range(1, len(face)):
        sum = np.add(sum, face[i])
    center = sum / float(len(face))
    return center


def is_facing(center, eye, normal):
    return np.dot(np.subtract(-center, eye), normal) > 0


def point_to_pixel(camera, raw_p):
    p = np.array(raw_p + [1])

    near_plane_dist = camera['near_plane_dist']
    near_plane_width = camera['near_plane_width']
    near_plane_height = camera['near_plane_height']
    p_cam = np.dot(p, camera['cam_mat'])

    if -p_cam[2] < near_plane_dist:
        return None

    canv_x = -near_plane_dist * p_cam[0] / p_cam[2]
    canv_y = -near_plane_dist * p_cam[1] / p_cam[2]
    screen_width = 800
    screen_height = 600
    ndc_x = canv_x/near_plane_width + 0.5
    ndc_y = canv_y/near_plane_height + 0.5
    x = ndc_x * screen_width
    y = ndc_y * screen_height

    if x > screen_width or y > screen_height or x < 0 or y < 0:
        return None

    return (x, y)


def get_camera(pos, z_axis):
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


def parse_camera_file(file_name):
    with open(file_name) as file:
        lines = filter(
            lambda x: not x.startswith('#') and len(x) != 0,
            [l.rstrip('\n') for l in file.readlines()]
        )

        if len(lines) < 3:
            sys.exit('Bad camera file %s' % file_name)

        near_plane_info = parse_point_list(lines[0])[0]
        pos = parse_point_list(lines[1])[0]
        z_axis = parse_point_list(lines[2])[0]
        if near_plane_info is None:
            sys.exit('Bad camera file %s' % file_name)
        if pos is None or z_axis is None:
            sys.exit('Bad camera file %s' % file_name)

        return {
            'near_plane_dist': near_plane_info[0],
            'near_plane_width': near_plane_info[1],
            'near_plane_height': near_plane_info[2],
            'eye': pos,
            'at': z_axis,
            'cam_mat': get_camera(pos, z_axis)
        }
    sys.exit('Bad camera file %s' % file_name)


def parse_point_list(points):
    try:
        return [
            [float(n), float(m), float(p)] for n, m, p in [
                s.split(',') for s in points.split('|')
            ]
        ]
    except:
        return None


def get_box(pos, l, w, h, color=False, outline=False):
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

        faces.append((
            [points[x] for x in face_by_points],
            color,
            outline
        ))

    return {'faces': faces}


def merge_models(models):
    model = {'faces': []}
    for submodel in models:
        model['faces'] += submodel['faces']
    return model

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: %s CAMERA_FILE_NAME' % sys.argv[0])

    camera = parse_camera_file(sys.argv[1])
    box1 = get_box(
        [0, 0, 0],
        40, 40, 40,
        color=(255, 0, 0),
        outline=False
    )
    box2 = get_box(
        [-70, 20, 10],
        40, 40, 40,
        color=(255, 0, 0),
        outline=False
    )
    print 'fuck'
    print get_lit_color((255,0,0), [1,0,0])
    model = merge_models([box1, box2])
    render(camera, model)
