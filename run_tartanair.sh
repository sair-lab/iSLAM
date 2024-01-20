# export CUDA_VISIBLE_DEVICES=1

data_dir=/data/tartanair/ocean/Hard/P000

loss_weight='(1.5,0.125,1.6875,0.025)'
rot_w=1
trans_w=0.1
batch_size=8
lr=3e-6
epoch=14
start_epoch=1

project_name=test_tartanair
train_name=exp_bs=${batch_size}_lr=${lr}_lw=${loss_weight}_${exp_type}

mkdir -p train_results/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}


python train.py \
    --result-dir train_results/${project_name}/${train_name} \
    --save-model-dir train_results_models/${project_name}/${train_name} \
    --project-name ${project_name} \
    --train-name ${train_name} \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --imu-denoise-model-name ./models/1022_tartanair_all_len80_10_1_0_direct_supervise_epoch_210_train_loss_0.001068338142439274.pth \
    --batch-size ${batch_size} \
    --worker-num 2 \
    --data-root ${data_dir} \
    --start-frame 0 \
    --end-frame -1 \
    --train-epoch ${epoch} \
    --start-epoch ${start_epoch} \
    --print-interval 1 \
    --snapshot-interval 10 \
    --lr ${lr} \
    --loss-weight ${loss_weight} \
    --data-type tartanair \
    --fix-model-parts 'flow' 'stereo' \
    --rot-w ${rot_w} \
    --trans-w ${trans_w}
