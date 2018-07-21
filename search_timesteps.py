# coding: utf-8
import numpy as np
import csv

import utils.config_util as conf
import model_object as mo

original_seq = np.loadtxt(conf.get_filename_via_tpl('seq', n_users=12, n_samples=2192), delimiter=',')
print(original_seq.shape)

times_steps = {original_seq.shape[1]}
results = [['time_steps', 'obj_func', 'h_seq', 'te_mat', 'complexity', 'is_max']]
while len(times_steps) < original_seq.shape[1]:
    max_obj = max_idx = -1
    for i in range(1, original_seq.shape[1]):
        if i not in times_steps:
            temp_steps = [i] + list(times_steps)
            temp_steps.sort()

            obj, h_seq, te_mat, complexity = mo.cal_object_function(original_seq, temp_steps, return_details=True)
            line = [temp_steps, obj, h_seq, te_mat, complexity]
            line = [str(i) for i in line]

            if obj > max_obj:
                times_steps.add(i)
                max_obj = obj
                max_idx = i
                line.append('True')
            else:
                line.append('')
            results.append(line)

    print(max_obj, max_idx)

with open('search_timesteps_results_no_permutationtest.csv', 'a', newline='') as fp:
    csv_writer = csv.writer(fp)
    csv_writer.writerows(results)






