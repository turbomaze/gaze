import numpy as np
from context import Rasta, Scene
from Tkinter import Label
from PIL import Image, ImageDraw, ImageFilter, ImageTk


class GazeProblem(object):
    def __init__(
        self, root, dims, num_boxes, radius
    ):
        self.root = root
        self.dims = dims
        self.num_boxes = num_boxes
        self.radius = radius

        # get the renderer
        camera_file = '../data/camera.cam'
        global_offset = [-84., -62., 0.]
        global_scale = [12.6, 12.6, 10.]
        self.rasta = Rasta(
            camera_file,
            global_offset, global_scale,
            dims[0], dims[1]
        )

    def render(self, img, x):
        img = self.get_image(x).filter(
            ImageFilter.GaussianBlur(radius=self.radius)
        )
        draw = ImageDraw.Draw(img)
        draw.text((30, 10), str([round(c, 3) for c in x]), fill="#000000")
        tk_img = ImageTk.PhotoImage(img)
        label_image = Label(self.root, image=tk_img)
        label_image.place(
            x=0, y=0,
            width=img.size[0],
            height=img.size[1]
        )
        self.root.update()

    def get_image(self, latent):
        model = Scene.get_model_from_latent(latent)
        img = self.rasta.render(model)
        return img

    def get_random(self):
        return Scene.sample(self.num_boxes)

    def get_next(self, latent, k, factor):
        # TODO
        return self.get_random()

    def get_likelihood_func(self, goal_img):
        small_size = (100, 75)
        small_goal_img = goal_img.resize(
            small_size, Image.BILINEAR
        ).filter(
            ImageFilter.GaussianBlur(radius=self.radius)
        )
        data_a = np.array(small_goal_img.getdata())

        def get_likelihood(x):
            b = self.get_image(x).resize(
                small_size, Image.BILINEAR
            ).filter(
                ImageFilter.GaussianBlur(radius=self.radius)
            )
            data_b = np.array(b.getdata())

            # direct error
            a_sub_b = np.subtract(data_a, data_b)
            diff = np.linalg.norm(
                a_sub_b[:, 0]
            ) + np.linalg.norm(
                a_sub_b[:, 1]
            ) + np.linalg.norm(
                a_sub_b[:, 2]
            )
            return 1./diff

        return get_likelihood

    def get_prior_prob(self, x):
        return 1.
