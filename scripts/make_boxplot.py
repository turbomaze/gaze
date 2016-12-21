import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import cPickle

if len(sys.argv) < 2:
    print 'Must provide a comma separated list of experiments.'
    sys.exit(0)

results_base = 'results-'
root_dir = '../data/experiments/experiment-'
experiment_ids = sys.argv[1].split(',')

data = [
    cPickle.load(open('%s%s/%s%s.p' % (
        root_dir, id, results_base, id
    ), 'rb'))
    for id in experiment_ids
]

scores = [[
    result['guess_score'] for result in experiment
] for experiment in data]
num_samples = [
    experiment[0]['num_samples']
    for experiment in data
]


font = {'size': 22}
rc('font', **font)
fig = plt.figure(1)
ax = fig.add_subplot(111)
bp = ax.boxplot(scores)

ax.set_xticklabels(num_samples)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.title('Real loss vs. # samples (with elitism)')
plt.xlabel('# samples')
plt.ylabel('Real loss')
plt.show()
