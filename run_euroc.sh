# export CUDA_VISIBLE_DEVICES=1

data_dir=/data/euroc/MH_01_easy/mav0

loss_weight='(4,0.1,2,0.1)'
rot_w=1
trans_w=0.1
batch_size=8
lr=3e-6
epoch=14
start_epoch=1

project_name=test_euroc
train_name=exp_bs=${batch_size}_lr=${lr}_lw=${loss_weight}

mkdir -p train_results/${project_name}/${train_name}
mkdir -p train_results_models/${project_name}/${train_name}


python train.py \
    --result-dir train_results/${project_name}/${train_name} \
    --save-model-dir train_results_models/${project_name}/${train_name} \
    --project-name ${project_name} \
    --train-name ${train_name} \
    --vo-model-name ./models/stereo_cvt_tartanvo_1914.pkl \
    --imu-denoise-model-name ./models/1029_euroc_no_cov_1layer_epoch_100_train_loss_0.19208121810994155.pth \
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
    --data-type euroc \
    --fix-model-parts 'flow' 'stereo' \
    --rot-w ${rot_w} \
    --trans-w ${trans_w}
