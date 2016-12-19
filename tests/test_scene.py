from context import Rasta, Scene


num_boxes = 3

# get the renderer
camera = '../data/camera.cam'
out_file = '../out.png'
global_offset = [-84., -62., 0.]
global_scale = [12.6, 12.6, 10.]
rasta = Rasta(camera, global_offset, global_scale, 800, 600)

# get the scene model
latent = Scene.sample(num_boxes)
model = Scene.get_model_from_latent(latent)

# render the scene
rasta.render_to_image(model, out_file)
