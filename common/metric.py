import numpy as np
from collections import defaultdict
import sklearn.metrics as metrics
import pdb
CROSS_VAL = {1:3,2:2,3:1}
GROUND_TRUTH = {
    1:['xxxxxxx_grade_1',
       'xxxxxxx_grade_2',
       ],

    2:[
       'xxxxxxx_grade_1',
       'xxxxxxx_grade_2',
],
  3:['xxxxxxx_grade_1',
     'xxxxxxx_grade_2',
]

}
class ImgLevelResult(object):
    def __init__(self, args):
        self.args = args
        self.imglist = {}
        for name in GROUND_TRUTH[CROSS_VAL[args.cross_val]]:
            # get the ground truth label: noral-0, low-level:1, high-level:2
            self.imglist[name.split('_grade')[0]] = int(name.split('_')[-1])-1
        self.prediction = defaultdict(list)

    def patch_result(self, name, label):
        name = name.split('/')[-1].split('_grade')[0]
        self.prediction[name].append(label)

    def batch_patch_result(self, names, labels):
        for name, label in zip(names, labels):
            name = name.split('/')[-1].split('_grade')[0]
            self.prediction[name].append(label)

    def final_result(self):
        result = []
        gt = []
        binary_result = []
        binary_gt = []
        for key, value in self.prediction.items():
            gt.append(self.imglist[key])
            binary_gt.append(1 if self.imglist[key]>0 else 0)
            _result = [value.count(0), value.count(1), value.count(2)]
            result.append(np.argmax(_result))
            binary_result.append(1 if np.argmax(_result)>0 else 0)
        acc = metrics.accuracy_score(gt, result)
        binary_acc = metrics.accuracy_score(binary_gt,binary_result)
        return acc,binary_acc