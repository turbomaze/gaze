import numpy as np
from Tkinter import Tk
from context import Scene, core, MH

def get_real_loss(a, b):
    diff = np.subtract(a, b)
    sq = np.square(diff)
    return np.sum(sq) ** 0.5

num_trials = 500
dims = (400, 300)
root = Tk()
root.geometry(str(dims[0]) + 'x' + str(dims[1]))

# domain specific
num_boxes = 2
correct = Scene.sample(num_boxes)
problem = core.GazeProblem(root, dims, num_boxes, radius=20)
correct_img = problem.get_image(correct)

first_guess = Scene.sample(num_boxes)
print 'First guess: ', first_guess
print 'First score: ', get_real_loss(first_guess, correct)
metropolis = MH(
    problem.get_next,
    problem.get_likelihood_func,
    problem.get_prior_prob,
    lambda x: problem.render(problem.get_image(x), x)
)
guess = metropolis.optimize(
    correct_img, first_guess, trials=num_trials
)

print 'Guess: ', guess
print 'Score: ', get_real_loss(guess, correct)
