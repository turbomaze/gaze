"""
GPGP
@author Anthony Liu <igliu@mit.edu>
@version 1.0.1
"""

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
        img = self.get_image(x)
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

    def get_next(self, latent, k):
        return Scene.transition(latent, k)

    def get_blob_likelihood_func(self, goal_img):
        # get the blobs for the image
        small_size = (200, 150)
        small_goal_img = goal_img.resize(
            small_size, Image.BILINEAR
        )
        small_arr = np.array(small_goal_img)
        colorgrams = self.get_colorgrams(small_arr)
        blobs = self.get_blobs_from_colorgrams(colorgrams)

        def get_likelihood(x):
            small_guess_img = self.get_image(x).resize(
                small_size, Image.BILINEAR
            )
            guess_arr = np.array(small_guess_img)
            guess_colorgrams = self.get_colorgrams(guess_arr)
            guess_blobs = self.get_blobs_from_colorgrams(
                guess_colorgrams
            )
            diff = self.get_blobs_diff(blobs, guess_blobs)
            return 1./diff

        return get_likelihood

    @classmethod
    def get_blobs_diff(cls, blobs_a, blobs_b):
        diff = 0
        for color in blobs_a:
            # color is in in both blobs
            if color in blobs_b:
                diff += cls.get_blob_diff(
                    blobs_a[color]['x'], blobs_b[color]['x']
                )
                diff += cls.get_blob_diff(
                    blobs_a[color]['y'], blobs_b[color]['y']
                )

            # color is in a but not in b
            else:
                diff += np.sum(abs(blobs_a[color]['x'])) ** 2
                diff += np.sum(abs(blobs_a[color]['y'])) ** 2

        # color is not in b
        for color in blobs_b:
            if color not in blobs_a:
                diff += np.sum(abs(blobs_b[color]['x'])) ** 2
                diff += np.sum(abs(blobs_b[color]['y'])) ** 2

        return diff

    @classmethod
    def get_blob_diff(cls, blob_a, blob_b):
        if len(blob_a) > len(blob_b):
            return cls.get_blob_diff(blob_b, blob_a)

        a, b = list(blob_a), np.array(list(blob_b))
        diff = 0
        for i, x in enumerate(a):
            # find closest in b
            dist_b = abs(np.array(b) - x)
            min_idx = np.argmin(dist_b)

            # add the error and remove the chosen element
            diff += dist_b[min_idx]
            b = np.delete(b, min_idx)

        # distance from 0 for unselected values
        diff += np.sum(abs(b))
        return diff

    def get_colorgrams(self, arr):
        colorgrams = {}

        arr_dims = arr.shape
        for x in range(arr_dims[1]):
            for y in range(arr_dims[0]):
                color = arr[y][x]
                c_str = ','.join(map(str, color))

                # skip white, black, and gray
                if c_str == '0,0,0':
                    continue
                elif c_str == '255,255,255':
                    continue
                elif color[0] == color[1]:
                    if color[1] == color[2]:
                        continue

                if c_str not in colorgrams:
                    colorgrams[c_str] = {}
                    colorgrams[c_str]['x'] = [0]*arr_dims[1]
                    colorgrams[c_str]['y'] = [0]*arr_dims[0]

                colorgrams[c_str]['x'][x] += 1
                colorgrams[c_str]['y'][y] += 1

        return colorgrams

    @classmethod
    def get_blobs_from_colorgrams(cls, colorgrams):
        blobs = {}
        for color in colorgrams:
            colorgram = colorgrams[color]
            x_avgs = cls.get_blobs_from_list(colorgram['x'])
            y_avgs = cls.get_blobs_from_list(colorgram['y'])
            blobs[color] = {
                'x': x_avgs,
                'y': y_avgs,
            }
        return blobs

    @classmethod
    def get_blobs_from_list(cls, list):
        avgs = []
        prev_count = 0
        streak_sum = 0
        streak_count = 0
        for i, count in enumerate(list):
            if count == 0 and count != prev_count:
                avgs += [streak_sum/streak_count]
                streak_sum = 0
                streak_count = 0
            elif count != 0:
                streak_sum += i * count
                streak_count += count
            prev_count = count

        if prev_count != 0:
            avgs += [streak_sum/streak_count]

        return np.array(avgs)

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

    def get_prior_prob(self, latent):
        def eval_gaussian(mu, stdev, x):
            coeff = 1./(stdev * (2*np.pi)**0.5)
            exp_coeff = -1./(2 * stdev**2)
            return coeff * np.e ** (exp_coeff * (x - mu)**2)

        prob = 1.
        for i, z in enumerate(latent):
            if i == 0:
                prob *= eval_gaussian(
                    0.75 * Scene.x_range,
                    Scene.x_range/16.,
                    z
                )
            elif i == 1:
                prob *= eval_gaussian(
                    Scene.z_range/2.,
                    Scene.z_range/4.,
                    z
                )
            elif i == 2:
                prob *= 1./self.num_boxes
            else:
                idx_in_box = i - Scene.num_viewer_params
                idx_in_box = idx_in_box % Scene.num_box_params
                if idx_in_box == 0:
                    prob *= 1./(Scene.x_range/2. - 1.)
                elif idx_in_box == 0:
                    prob *= 1./(Scene.y_range - 1.)
                elif idx_in_box == 0:
                    prob *= 1./(Scene.z_range - 1.)
        return prob
