# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:35:56 2019

@author: ndo
"""
import basic.processor as WP
import basic.misc as misc
import matplotlib.pyplot as plt
import numpy as np

plt.close("all")


def plot_this(arg):
    x, y, data = arg
    #    print(xy, np.shape(data))
    plt.plot(data, label=f"{x},{y}")


def my_gen(our_array):
    len_z, len_x, len_y = np.shape(our_array)
    for x in range(len_x):
        for y in range(len_y):
            yield [x, y, our_array[:, x, y]]


test_options = misc.json_to_dict("test_data\\test_low_field_options.json")

wp = WP.Processor(test_options)

wp.process_file()

b = wp.dw.sig_norm
# for d in my_gen(b):
#    plt.plot(d)

# a = [(x, y) for x,y in np.ndindex((32,32))]

c = my_gen(b)
# print(a)
# print(np.shape(np.array(np.ndindex((32,32)))))
assyi = map(plot_this, c)
#
plt.figure("Hi")
for a in assyi:
    pass
