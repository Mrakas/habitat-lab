import numpy as np

random_heading = 1.308996893378039
random_rotation = [
        0,
        -np.sin(random_heading / 2),
        0,
        -np.cos(random_heading / 2),
    ]
# rotation to heading
print(random_rotation)

'''
"heading": -2.6179938738007547,
"start_rotation": [-0.0, 0.9659258262890683, -0.0, -0.25881904510252063],
'''