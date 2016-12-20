from Tkinter import Tk
from context import Scene, core, MH

num_samples = 250
dims = (200, 150)
root = Tk()
root.geometry(str(dims[0]) + 'x' + str(dims[1]))

# domain specific
num_boxes = 3
correct = Scene.sample(num_boxes)
problem = core.GazeProblem(root, dims, num_boxes, radius=14)
correct_img = problem.get_image(correct)
correct_img.save('correct.png')

first_guess = Scene.sample(num_boxes)
print 'Correct: ', map(lambda x: round(x, 1), correct)
print 'First guess: ', map(lambda x: round(x, 1), first_guess)
print 'First score: ', Scene.get_target_loss(
    first_guess, correct
)
metropolis = MH(
    problem.get_next,
    problem.get_likelihood_func,
    problem.get_prior_prob,
    lambda x: problem.render(problem.get_image(x), x)
)
guess = metropolis.optimize(
    correct_img, first_guess, samples=num_samples, do_log=True
)

print 'Guess: ', map(lambda x: round(x, 1), guess)
print 'Score: ', Scene.get_target_loss(guess, correct)
