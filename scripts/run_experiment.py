import os
import sys
import time
import numpy as np
import cPickle
from context import Scene, core, MH

if len(sys.argv) < 2:
    print 'You must specify the root experiments directory.'
if len(sys.argv) < 3:
    print 'Must supply number of boxes'
if len(sys.argv) < 4:
    print 'Must supply number of rounds'
if len(sys.argv) < 5:
    print 'Must supply number of samples (per inference)'
if len(sys.argv) < 6:
    print 'Must supply whether or not to use elitism'
    sys.exit(0)

start = time.clock()

# parameters
root_dir = sys.argv[1]
num_boxes = int(sys.argv[2])
num_rounds = int(sys.argv[3])
num_samples = int(sys.argv[4])
use_elitism = int(sys.argv[5]) == 1
dims = (200, 150)
experiment_id = np.base_repr(int(9**4 * np.random.rand()), 36)

# create a directory to save all the data
dir_name = '%s/experiment-%s' % (root_dir, experiment_id)
target_dir_name = '%s/target-imgs' % dir_name
guess_dir_name = '%s/guess-imgs' % dir_name
os.makedirs(dir_name)
os.makedirs(target_dir_name)
os.makedirs(guess_dir_name)

# set up the problem
problem = core.GazeProblem(None, dims, num_boxes, radius=14)
metropolis = MH(
    problem.get_next,
    problem.get_likelihood_func,
    problem.get_prior_prob,
    lambda x: x,
    elite=use_elitism
)

# run through all the rounds
results = []
for i in range(num_rounds):
    print '\nROUND %d' % i
    target = Scene.sample(num_boxes)
    target_img = problem.get_image(target)
    target_name = '%s/target-%d.png' % (target_dir_name, i)
    target_img.save(target_name)
    print 'Target: ', map(lambda x: round(x, 1), target)

    first_guess = Scene.sample(num_boxes)
    print 'First score: %f' % Scene.get_target_loss(
        first_guess, target
    )
    print 'Initial:', map(lambda x: round(x, 1), first_guess)

    guess = metropolis.optimize(
        target_img, first_guess, samples=num_samples
    )
    guess_img = problem.get_image(guess)
    guess_name = '%s/guess-%d.png' % (guess_dir_name, i)
    guess_img.save(guess_name)
    guess_score = Scene.get_target_loss(guess, target)
    print 'Final:  ', map(lambda x: round(x, 1), guess)
    print 'Guess score: %f' % guess_score
    results.append({
        'num_samples': num_samples,
        'target': target,
        'guess': guess,
        'guess_score': guess_score
    })

# pickle the results array to the experiments dir
print 'Saving results array to the experiments folder...'
results_name = '%s/results-%s.p' % (dir_name, experiment_id)
cPickle.dump(results, open(results_name, 'wb'))

# report overall stats
duration = time.clock() - start
average_score = sum(r['guess_score'] for r in results)
average_score /= float(len(results))
print '\nExperiment %s: %d rounds of %d samples in %fs' % (
    experiment_id, num_rounds, num_samples, duration
)
print 'Average score: %f' % average_score
