import sys
import math
import numpy as np
from PIL import Image, ImageDraw


def render(camera, model):
    data = np.zeros((600, 800, 3), dtype=np.uint8)
    data.fill(255)
    img = Image.fromarray(data, 'RGB')
    draw = ImageDraw.Draw(img, 'RGB')
    # print model
    for triangle, color, outline in model['triangles']:
        points = filter(
            lambda x: x is not None,
            map(point_to_pixel, triangle)
        )
        if len(points) < 3:
            continue
        if outline:
            draw.polygon(points, outline=color)
        else:
            draw.polygon(points, fill=color)
    del draw
    img.save('out.png', 'PNG')


def point_to_pixel(raw_p):
    p = np.array(raw_p + [1])
    cam_space = get_camera([10, 10, 10.], [0, 0, -1])

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


def get_box(pos, l, w, h, color=False, outline=False):
    triangles = []
    points = [
        [i >> 2, (i >> 1) & 1, i & 1] for i in range(2**3)
    ]
    points = [
        [l*a + pos[0], w*b + pos[1], h*c + pos[2]]
        for a, b, c in points
    ]
    faces = [
        [0, 1, 3, 2],  # x-axis
        [4, 5, 7, 6],  # x-axis
        [0, 1, 5, 4],  # y-axis
        [2, 3, 7, 6],  # y-axis
        [0, 2, 6, 4],  # z-axis
        [1, 3, 7, 5]  # z-axis
    ]
    m = 20

    for face in faces:
        base_color = tuple([
            max(min(int(np.random.uniform() * 256), 255), 0)
            for _ in range(3)
        ])
        if not (color is False):
            base_color = color
        color1 = tuple([
            max(min(int(a + m), 255), 0)
            for a in base_color
        ])
        color2 = tuple([
            max(min(int(a - m), 255), 0)
            for a in base_color
        ])

        pa, pb, pc, pd = [points[x] for x in face]
        triangles.append(([pa, pb, pc], color1, outline))
        triangles.append(([pa, pc, pd], color2, outline))

    return {'triangles': triangles}


def merge_models(models):
    model = {'triangles': []}
    for submodel in models:
        model['triangles'] += submodel['triangles']
    return model

if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('Usage: %s CAMERA_FILE_NAME' % sys.argv[0])

    camera = parse_camera_file(sys.argv[1])
    box1 = get_box(
        [0, 0, 0],
        40, 40, 40,
        color=(255, 0, 0),
        outline=True
    )
    box2 = get_box(
        [-70, 20, 10],
        40, 40, 40,
        color=(255, 0, 0),
        outline=True
    )
    model = merge_models([box1, box2])
    render(camera, model)
