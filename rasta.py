import sys
import numpy as np
import random as r
from PIL import Image, ImageDraw


def render(camera, model):
    print model
    data = np.zeros((600, 800, 3), dtype=np.uint8)
    data.fill(255)
    img = Image.fromarray(data, 'RGB')
    draw = ImageDraw.Draw(img, 'RGB')
    for triangle in model['triangles']:
        points = map(point_to_pixel, triangle)
        col = int(256 * r.random())
        draw.polygon(points, (col, col, col))
    del draw
    img.save('out.png', 'PNG')

def point_to_pixel(raw_p):
    p = np.array(raw_p + [1])
    cam_space = np.array([
        [1., 0.,  0., 0.],
        [0., 1.,  0., 0.],
        [0., 0., -1., 0.],
        [0., 0.,  0., 1.]
    ])
    near_plane_dist = 1.
    near_plane_width = 160.
    near_plane_height = 120.
    p_cam = np.dot(cam_space, p)
    canv_x = -near_plane_dist * p_cam[0] / p_cam[2]
    canv_y = -near_plane_dist * p_cam[1] / p_cam[2]
    screen_width = 800
    screen_height = 600
    ndc_x = canv_x/near_plane_width + 0.5
    ndc_y = canv_y/near_plane_height + 0.5
    x = ndc_x * screen_width
    y = ndc_y * screen_height
    return (x, y)


def parse_graphics_file(file_name):
    with open(file_name) as file:
        lines = [l.rstrip('\n') for l in file.readlines()]
        model = {}
        model['triangles'] = []
        for l in lines:
            if not l.startswith('#') and len(l) > 0:
                triangle = parse_point_list(l)
                if triangle is None:
                    sys.exit('Bad model file %s' % file_name)
                model['triangles'].append(triangle)
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