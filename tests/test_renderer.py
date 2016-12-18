from context import Rasta

camera = '../data/camera.cam'
out_file = '../out.png'
global_offset = [-84., -62., 0.]
global_scale = [12.6, 12.6, 10.]
rasta = Rasta(camera, global_offset, global_scale, 800, 600)

boundary = Rasta.get_box(
    [0, 0, 0],
    13.3333, 10, 10,
    color=(0, 0, 0),
    outline=True
)
models = [boundary]
models.append(Rasta.rotate_box(Rasta.get_box(
    [10, 7, 5],
    1, 1, 1,
    color=(255, 0, 0),
    outline=False
), 3.14/3))

rasta.render(models, out_file)
