import sys
import math
import numpy as np
from PIL import Image, ImageDraw


def render(camera, model):
    data = np.zeros((600, 800, 3), dtype=np.uint8)
    data.fill(255)
    img = Image.fromarray(data, 'RGB')
    draw = ImageDraw.Draw(img, 'RGB')
    for triangle, color in model['triangles']:
        points = filter(
            lambda x: x is not None,
            map(point_to_pixel, triangle)
        )
        if len(points) < 3:
            continue
        draw.polygon(points, color)
    del draw
    img.save('out.png', 'PNG')


def point_to_pixel(raw_p):
    p = np.array(raw_p + [1])
    cam_space = get_camera([0, 0., -15.], [0, 0, -1])

    near_plane_dist = 5.
    near_plane_width = 160.
    near_plane_height = 120.
    p_cam = np.dot(p, cam_space)

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
    print y_rot, x_rot
    print cam_mat
    return np.linalg.inv(cam_mat)


def parse_graphics_file(file_name):
    with open(file_name) as file:
        lines = [l.rstrip('\n') for l in file.readlines()]
        model = {}
        model['triangles'] = []
        for l in lines:
            if not l.startswith('#') and len(l) > 0:
                parts = l.split(':')
                triangle = parse_point_list(parts[0])
                color = tuple(
                    int(x) for x in parse_point_list(
                        parts[1]
                    )[0]
                )
                if triangle is None or color is None:
                    sys.exit('Bad model file %s' % file_name)
                model['triangles'].append((triangle, color))
        return model
    sys.exit('Bad model file %s' % file_name)


def parse_camera_file(file_name):
    with open(file_name) as file:
        lines = filter(
            lambda x: not x.startswith('#') and len(x) != 0,
            [l.rstrip('\n') for l in file.readlines()]
        )

        if len(lines) < 2:
            sys.exit('Bad camera file %s' % file_name)

        near_plane = parse_point_list(lines[0])
        far_plane = parse_point_list(lines[1])
        if near_plane is None or far_plane is None:
            sys.exit('Bad camera file %s' % file_name)

        return {
            'near_plane': near_plane,
            'far_plane': far_plane,
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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: %s CAMERA_FILE_NAME MODEL_FILE_NAME' % sys.argv[0])

    camera = parse_camera_file(sys.argv[1])
    model = parse_graphics_file(sys.argv[2])
    render(camera, model)
