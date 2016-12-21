import sys
import cPickle
from context import Scene

if len(sys.argv) < 2:
    print 'You must specify the results file.'

results_name = sys.argv[1]
results = cPickle.load(open(results_name, 'rb'))

summaries = map(lambda result: {
    'target': map(
        lambda x: round(x, 1),
        Scene.get_target_box(result['target'])
    ),
    'guess': map(
        lambda x: round(x, 1),
        Scene.get_target_box(result['guess'])
    ),
    'score': round(result['guess_score'], 2)
}, results)

print 'Num samples: %d' % results[0]['num_samples']
print 'Num rounds: %d' % len(results)
for summary in summaries:
    print summary

