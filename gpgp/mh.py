"""
GPGP
@author Anthony Liu <igliu@mit.edu>
@version 1.0.1
"""

import random as rand
import time


class MH(object):
    # @param G -------- proposes a new state to transition to
    # @param pi_maker -  the post likelihood function maker
    # @param q -------- prior prob of the latent variables
    def __init__(self, G, pi_maker, q, progress, elite=True):
        self.elite = elite
        self.G = G
        self.pi_maker = pi_maker
        self.q = q
        self.progress = progress

    def optimize(self, goal_img, x_0, samples, do_log=False):
        start = time.clock()
        pi = self.pi_maker(goal_img)

        # Initialization: pick an initial state x at random
        x = x_0
        self.progress(x)
        prior_x = self.q(x)
        post_x = pi(x)

        # keep track of the most likely state
        x_max = x
        post_max = post_x
        for i in range(samples):
            if do_log and i % 5 == 0:
                print i, round(post_max, 6), map(
                    lambda x: round(x, 1),
                    x_max
                )

            k = i % len(x)
            xp = self.G(x, k)

            # compute the prior probability of x'
            prior_xp = self.q(xp)

            # render x' into I_r' to compute pi(x')
            post_xp = pi(xp)
            if post_xp > post_max:
                x_max = xp
                post_max = post_xp
            self.progress(xp)

            # set A(x'|x)=pi(I_r')/pi(I_r) * q(x')/q(x)
            acceptance = prior_xp/prior_x
            acceptance *= post_xp/post_x

            # accept the state according to A(x'|x).
            if self.elite:
                if post_xp > post_x:
                    # accept
                    x, prior_x, post_x = xp, prior_xp, post_xp
            else:
                accepted = rand.random() < acceptance
                if accepted or post_xp > post_x:
                    # accept
                    x, prior_x, post_x = xp, prior_xp, post_xp

        end = time.clock()
        duration = end - start
        if do_log:
            print '%d samples in %fs' % (samples, duration)
        return x_max
