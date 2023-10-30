import os

data_root = '/data/tartanair'

data_name = [
    'soulcity_Hard_P000',
    'soulcity_Hard_P001',
    'soulcity_Hard_P002',
    'soulcity_Hard_P003',
    'soulcity_Hard_P004',
    'soulcity_Hard_P005',
    'soulcity_Hard_P008',
    'soulcity_Hard_P009',
    # 'japanesealley_Hard_P000', 
    # 'japanesealley_Hard_P001', 
    # 'japanesealley_Hard_P002', 
    # 'japanesealley_Hard_P003', 
    # 'japanesealley_Hard_P004', 
    # 'japanesealley_Hard_P005', 
    'ocean_Hard_P000',
    'ocean_Hard_P001',
    'ocean_Hard_P002',
    'ocean_Hard_P003',
    'ocean_Hard_P004',
    'ocean_Hard_P005',
    'ocean_Hard_P006',
    'ocean_Hard_P007',
    'ocean_Hard_P008',
    'ocean_Hard_P009'
]

# data_name = [
#     'soulcity_Easy_P000',
#     'soulcity_Easy_P001',
#     'soulcity_Easy_P002',
#     'soulcity_Easy_P003',
#     'soulcity_Easy_P004',
#     'soulcity_Easy_P005',
#     'soulcity_Easy_P006',
#     'soulcity_Easy_P007',
#     'soulcity_Easy_P008',
#     'soulcity_Easy_P009',
#     'soulcity_Easy_P010',
#     'soulcity_Easy_P011',
#     'soulcity_Easy_P012',
#     'ocean_Easy_P000',
#     'ocean_Easy_P001',
#     'ocean_Easy_P002',
#     'ocean_Easy_P004',
#     'ocean_Easy_P005',
#     'ocean_Easy_P006',
#     'ocean_Easy_P008',
#     'ocean_Easy_P009',
#     'ocean_Easy_P010',
#     'ocean_Easy_P011',
#     'ocean_Easy_P012',
#     'ocean_Easy_P013'
# ]

# data_name = [
#     'seasidetown_Easy_P000',
#     'seasidetown_Easy_P001',
#     'seasidetown_Easy_P002',
#     'seasidetown_Easy_P003',
#     'seasidetown_Easy_P004',
#     'seasidetown_Easy_P005',
#     'seasidetown_Easy_P006',
#     'seasidetown_Easy_P007',
#     'seasidetown_Easy_P008',
#     'seasidetown_Easy_P009',
#     'seasidetown_Hard_P000',
#     'seasidetown_Hard_P001',
#     'seasidetown_Hard_P002',
#     'seasidetown_Hard_P004',
# ]

for dn in data_name:
    dir = data_root + '/' + dn.replace('_', '/')
    res_name = dn + '_alternative_80'

    cmd = "sh run_tartanair.sh {} {}".format(dir, res_name)

    print('\n>>>>>', cmd, '<<<<<\n')

    os.system(cmd)