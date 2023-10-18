import os

machine = '4090'

if machine == 'ccr':
    data_root = '/user/taimengf/projects/cwx/euroc'
elif machine == 'labserver':
    data_root = '/home/data2/euroc_raw'
elif machine == '4090':
    data_root = '/data/euroc'

data_name = [
    # 'MH_01_easy',
    'MH_02_easy',
    'MH_03_medium',
    'MH_04_difficult',
    'MH_05_difficult',
    'V1_01_easy',
    'V1_02_medium',
    'V1_03_difficult',
    'V2_01_easy',
    'V2_02_medium',
    'V2_03_difficult'
]

for dn in data_name:
    dir = data_root + '/' + dn + '/mav0'
    res_name = dn + '_optmbias'

    if machine == 'ccr':
        cmd = "sbatch run_euroc.sh {} {}".format(dir, res_name)
    else:
        cmd = "sh run_euroc.sh {} {}".format(dir, res_name)

    print('\n>>>>>', cmd, '<<<<<\n')

    os.system(cmd)
