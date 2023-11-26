import os

sh = '''
export CUDA_VISIBLE_DEVICES=2

data_dir=[DATA_DIR]

loss_weight='(1,0.1,10,0.1)'
rot_w=1
trans_w=0.1
batch_size=8
lr=3e-6
epoch=[EPOCH]
start_epoch=[START_EPOCH]
train_portion=1

exp_type='stereo'

project_name=[TRAIN_NAME]
train_name=exp_bs=${batch_size}_lr=${lr}_lw=${loss_weight}_${exp_type}

echo "=============================================="
echo "project name = ${project_name}"
echo "train name = ${train_name}"
echo "data dir = ${data_dir}"
echo "=============================================="

mkdir -p train_results/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}

# stereo: calc scale
python train.py \
    --result-dir train_results/${project_name}/${train_name} \
    --save-model-dir train_results_models/${project_name}/${train_name} \
    --project-name ${project_name} \
    --train-name ${train_name} \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --imu-denoise-model-name ./models/1025_kitti_no_cov_1layer_epoch_100_train_loss_3.092334827756494.pth \
    --batch-size ${batch_size} \
    --worker-num 2 \
    --data-root ${data_dir} \
    --start-frame 0 \
    --end-frame -1 \
    --train-epoch ${epoch} \
    --start-epoch ${start_epoch} \
    --print-interval 1 \
    --snapshot-interval 100 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --data-type kitti \
    --fix-model-parts 'flow' 'stereo' \
    --rot-w ${rot_w} \
    --trans-w ${trans_w} \
    --train-portion ${train_portion}
'''

# trainname = 'loop_train_kitti'
# data_name_train = [
#     '2011_10_03_drive_0042',
#     '2011_10_03_drive_0034',
#     '2011_09_30_drive_0016',
#     '2011_09_30_drive_0018',
#     '2011_09_30_drive_0020',
# ]

# data_name_test = [
#     '2011_10_03_drive_0027',
#     '2011_09_30_drive_0027',
#     '2011_09_30_drive_0028',
#     '2011_09_30_drive_0033',
#     '2011_09_30_drive_0034'
# ]

# 2
# trainname = 'loop_train_kitti_2'
# data_name_train = [
#     '2011_10_03_drive_0042',
#     '2011_09_30_drive_0016',
#     '2011_09_30_drive_0020',
#     '2011_10_03_drive_0027',
#     '2011_09_30_drive_0028',
# ]

# data_name_test = [
#     '2011_09_30_drive_0018',
#     '2011_10_03_drive_0034',
#     '2011_09_30_drive_0027',
#     '2011_09_30_drive_0033',
#     '2011_09_30_drive_0034'
# ]

# 3
# trainname = 'loop_train_kitti_3'
# data_name_train = [
#     '2011_09_30_drive_0034',
#     '2011_09_30_drive_0020',
#     '2011_10_03_drive_0027',#
#     '2011_09_30_drive_0028',#
#     '2011_09_30_drive_0033',
# ]

# data_name_test = [
#     '2011_09_30_drive_0016',
#     '2011_09_30_drive_0018',
#     '2011_10_03_drive_0034',
#     '2011_09_30_drive_0027',#
#     '2011_10_03_drive_0042',
# ]

# 4
trainname = 'loop_train_kitti_4'
data_name_train = [
    '2011_09_30_drive_0034',
    '2011_10_03_drive_0034',
    '2011_10_03_drive_0027',#
    '2011_09_30_drive_0028',#
    '2011_09_30_drive_0033',
]

data_name_test = [
    '2011_09_30_drive_0020',
    '2011_09_30_drive_0016',
    '2011_09_30_drive_0018',
    '2011_09_30_drive_0027',#
    '2011_10_03_drive_0042',
]

epoch = 1
ptr = 0
while epoch <= 100:
    dataname = data_name_train[ptr]
    datadir = f'/data/kitti/{dataname[:10]}/{dataname}_sync'
    cmd = sh.replace('[DATA_DIR]', datadir).replace('[START_EPOCH]', str(epoch)).replace('[EPOCH]', str(epoch)).replace('[TRAIN_NAME]', trainname)
    os.system(cmd)

    epoch += 1
    ptr = (ptr + 1) % len(data_name_train)

    if (epoch-1) % 5 == 0:
        for dataname in data_name_test:
            datadir = f'/data/kitti/{dataname[:10]}/{dataname}_sync'
            cmd = sh.replace('[DATA_DIR]', datadir).replace('[START_EPOCH]', str(epoch)).replace('[EPOCH]', str(epoch)).replace('[TRAIN_NAME]', trainname)
            os.system(cmd)
            resultdir = f'/home/tymon/iSLAM/train_results/{trainname}/exp_bs=8_lr=3e-6_lw=\(1,0.1,10,0.1\)_stereo'
            os.system(f'mv {resultdir}/{epoch} {resultdir}/{epoch-1}_test_{dataname}')

# epoch = 6
# for dataname in data_name_test:
#     datadir = f'/data/kitti/{dataname[:10]}/{dataname}_sync'
#     cmd = sh.replace('[DATA_DIR]', datadir).replace('[START_EPOCH]', str(epoch)).replace('[EPOCH]', str(epoch)).replace('[TRAIN_NAME]', trainname)
#     os.system(cmd)
#     resultdir = f'/home/tymon/iSLAM/train_results/{trainname}/exp_bs=8_lr=3e-6_lw=\(1,0.1,10,0.1\)_stereo'
#     os.system(f'mv {resultdir}/{epoch} {resultdir}/{epoch-1}_test_{dataname}')