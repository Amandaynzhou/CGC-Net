import numpy as np
from collections import defaultdict
import sklearn.metrics as metrics
import pdb
CROSS_VAL = {1:3,2:2,3:1}
GROUND_TRUTH = {
    1:['H08-144335_A1H&E_1_2_grade_1',
       'H08-144335_A1H&E_1_3_grade_1',
       'H08-57464_A1H&E_2_1_grade_1',
       'H09-23100_A1H_E_1_2_grade_1',
       'H09-12618_A1H_E_1_1_grade_1',
       'H09-18586_A2H&E_1_13_grade_1',
       'H08-144335_A1H&E_1_4_grade_1',
       'H09-00804_A2H_E_1_2_grade_1',
       'H08-50593_A1H&E_1_2_grade_1',
       'H09-11303_A2H&E_1_2_grade_1',
       'H08-36034_A1H&E_1_2_grade_1',
       'H09-00622_A2H_E_1_3_grade_1',
       'H09-00320_A2H&E_1_3_grade_1',
       'H09-18586_A2H&E_1_12_grade_1',
       'H09-00622_A2H_E_1_4_grade_1',
       'H09-02883_A2H_E_1_5_grade_1',
       'H09-00622_A2H&E_1_6_grade_1',
       'H09-12618_A1H_E_1_2_grade_1',
       'H09-16145_A2H_E_1_1_grade_1',
       'H10-24087_A2H_E_1_1_grade_1',
       'H08-19835_A1H&E_1_5_grade_1',
       'H09-18586_A2H_E_1_3_grade_1',
       'H09-00320_A2H_E_1_2_grade_1',
       'H08-68294_A1H&E_1_3_grade_1',
       'H06-02141_A3H_E_1_1_grade_2',
       'H06-05790_A4H_E_1_1_grade_2',
       'H09-00107_A1H_E_1_4_grade_2',
       'H09-00822_A1H_E_1_2_grade_2',
       'H09-00804_A2H_E_1_3_grade_2',
       'H06-04442_A5H_E_1_2_grade_2',
       'H06-04116_A6H_E_1_2_grade_2',
       'H09-00320_A2H_E_1_1_grade_2',
       'H09-00622_A2H_E_1_1_grade_2',
       'H06-05658_A5H_E_1_1_grade_2',
       'H06-05168_A5H_E_1_1_grade_2',
       'H09-16289_A1H_E_1_1_grade_3',
       'H09-02883_A2H_E_1_2_grade_3',
       'H09-18586_A2H_E_1_9_grade_3',
       'H09-18586_A2H_E_1_6_grade_3',
       'H09-23100_A1H_E_1_3_grade_3',
       'H09-24359_A2H_E_1_1_grade_3',
       'H09-24359_A2H_E_1_5_grade_3',
       'H09-18586_A2H_E_1_10_grade_3',
       'H09-15863_A2H_E_1_7_grade_3',
       'H09-02883_A2H_E_1_1_grade_3',
       'H09-15863_A2H_E_1_8_grade_3',],

    2:['H09-03067_A2H_E_1_1_grade_1',
       'H09-18586_A2H_E_1_8_grade_1',
       'H08-81054_A1H&E_1_3_grade_1',
       'H08-50593_A1H&E_1_3_grade_1',
       'H08-68294_A1H&E_1_9_grade_1',
       'H09-15863_A2H_E_1_1_grade_1',
       'H09-23100_A1H&E_1_5_grade_1',
       'H09-07547_A1H_E_1_1_grade_1',
       'H09-03067_A2H&E_1_2_grade_1',
       'H08-68294_A1H&E_1_1_grade_1',
       'H09-12618_A1H&E_1_3_grade_1',
       'H08-68294_A1H&E_1_2_grade_1',
       'H08-19835_A1H&E_1_3_grade_1',
       'H09-00622_A2H&E_1_5_grade_1',
       'H08-68294_A1H&E_1_11_grade_1',
       'H09-23100_A1H_E_1_1_grade_1',
       'H08-68294_A1H&E_1_4_grade_1',
       'H08-68294_A1H&E_1_5_grade_1',
       'H08-81054_A1H&E_1_2_grade_1',
       'H08-19835_A1H&E_1_4_grade_1',
       'H09-00107_A1H&E_1_7_grade_1',
       'H09-00107_A1H_E_1_2_grade_1',
       'H09-16145_A2H&E_1_3_grade_1',
       'H09-16145_A2H_E_1_2_grade_1',
       'H06-04255_A5H_E_1_1_grade_2',
       'H06-03838_A6H_E_1_1_grade_2',
       'H06-04442_A5H_E_1_3_grade_2',
       'H09-01265_A2H_E_1_1_grade_2',
       'H06-04442_A5H_E_1_1_grade_2',
       'H06-04255_A5H_E_1_3_grade_2',
       'H06-04116_A6H_E_1_1_grade_2',
       'H06-04767_B5H_E_1_1_grade_2',
       'H09-00822_A1H_E_1_3_grade_2',
       'H06-04767_B5H_E_1_2_grade_2',
       'H09-00622_A2H_E_1_2_grade_2',
       'H09-00804_A2H_E_1_1_grade_3',
       'H09-18586_A2H_E_1_5_grade_3',
       'H09-02883_A2H_E_1_6_grade_3',
       'H09-16289_A1H_E_1_2_grade_3',
       'H09-02883_A2H_E_1_4_grade_3',
       'H09-18586_A2H_E_1_4_grade_3',
       'H09-23100_A1H_E_1_4_grade_3',
       'H09-15863_A2H_E_1_6_grade_3',
       'H09-15863_A2H_E_1_2_grade_3',
       'H09-15863_A2H_E_1_3_grade_3',
       'H09-24359_A2H_E_1_6_grade_3',
       'H09-18586_A2H_E_1_7_grade_3',
],
  3:['H09-00107_A1H_E_1_1_grade_1',
     'H08-50593_A1H&E_1_4_grade_1',
     'H08-68294_A1H&E_1_8_grade_1',
     'H08-19835_A1H&E_1_2_grade_1',
     'H08-160413_A1H&E_1_1_grade_1',
     'H08-50593_A1H&E_1_1_grade_1',
     'H09-18586_A2H&E_1_11_grade_1',
     'H08-19835_A1H&E_1_1_grade_1',
     'H08-36034_A1H&E_1_1_grade_1',
     'H08-68294_A1H&E_1_10_grade_1',
     'H08-50593_A1H&E_1_5_grade_1',
     'H08-68294_A1H&E_1_7_grade_1',
     'H10-24087_A2H_E_1_2_grade_1',
     'H08-81054_A1H&E_1_1_grade_1',
     'H08-68294_A1H&E_1_6_grade_1',
     'H08-144335_A1H&E_1_1_grade_1',
     'H09-00622_A2H&E_1_7_grade_1',
     'H09-00107_A1H&E_1_6_grade_1',
     'H09-11303_A2H&E_1_3_grade_1',
     'H09-07547_A1H&E_1_2_grade_1',
     'H08-57464_A1H&E_2_2_grade_1',
     'H09-00107_A1H&E_1_5_grade_1',
     'H09-11303_A2H_E_1_1_grade_1',
     'H09-01648_A2H_E_1_2_grade_2',
     'H09-00822_A1H_E_1_1_grade_2',
     'H06-02141_A3H_E_1_3_grade_2',
     'H09-01648_A2H_E_1_1_grade_2',
     'H09-00107_A1H_E_1_3_grade_2',
     'H06-05869_A5H_E_1_1_grade_2',
     'H09-02464_A2H_E_1_1_grade_2',
     'H06-05869_A5H_E_1_2_grade_2',
     'H06-02141_A3H_E_1_2_grade_2',
     'H06-04255_A5H_E_1_2_grade_2',
     'H06-05658_A5H_E_1_2_grade_2',
     'H09-00804_A2H_E_1_4_grade_3',
     'H09-18586_A2H_E_1_2_grade_3',
     'H09-24359_A2H_E_1_4_grade_3',
     'H09-02883_A2H_E_1_3_grade_3',
     'H09-15863_A2H_E_1_5_grade_3',
     'H09-24359_A2H_E_1_3_grade_3',
     'H09-24359_A2H_E_1_2_grade_3',
     'H09-15863_A2H_E_1_4_grade_3',
     'H09-15863_A2H_E_1_9_grade_3',
     'H06-05790_A4H_E_1_2_grade_3',
     'H09-15863_A2H_E_1_10_grade_3',
     'H09-18586_A2H_E_1_1_grade_3',
]

}

class ImgLevelResult(object):
    def __init__(self, args):
        self.args = args
        self.imglist = {}
        for name in GROUND_TRUTH[CROSS_VAL[args.cross_val]]:

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