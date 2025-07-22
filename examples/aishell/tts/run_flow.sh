# Copyright 2025 Binbin Zhang(binbzha@qq.com)

[ ! -s west ] && ln -s ../../west
[ ! -s tools ] && ln -s ../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="0"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

llm=/bucket/output/jfs-hdfs/user/binbin.zhang/github/west/examples/aishell/tts/model/Qwen2.5-0.5B-Audio/
speech_tokenizer=/bucket/output/jfs-hdfs/user/binbin.zhang/models/s3tokenizer
speaker_model=/bucket/output/jfs-hdfs/user/binbin.zhang/models/wespeaker/campplus

stage=train # data/train/decode
data=data
dir=exp/Qwen2-130M-flow
steps=50000  # training steps


if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_type "flow_model" \
        --extractor_type "tts_flow" \
        --s3tokenizer_model_name_or_path $speech_tokenizer \
        --flow_llm_config_path $PWD/conf/flow_config.json \
        --speaker_model_path  $speaker_model \
        --text_tokenizer_path $llm \
        --data_path $data/train.list \
        --bf16 False \
        --batch_size 64 \
        --num_data_cycles 200 \
        --output_dir $dir \
        --max_steps $steps \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 100 \
        --learning_rate 3e-4 \
        --weight_decay 0.02 \
        --warmup_ratio 0.05 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "tensorboard" \
        --gradient_checkpointing \
        --dataloader_num_workers 2 \
        --dataloader_prefetch_factor 10 \
        --save_total_limit 10000 \
        --deepspeed conf/ds_config_zero3.json \
        --accelerator_config conf/accelerator_config.json
fi


if [ $stage == "decode" ] || [ $stage == "all" ]; then
    steps=10000
    mdir=$dir/checkpoint-${steps}
    tag=cfg0.7
    adir=$(echo $mdir | sed 's:exp:exp_audio:g')-${tag}
    mkdir -p $adir
    python west/bin/tts_flow_inference.py \
        --model_type "flow_model" \
        --extractor_type "tts_flow" \
        --s3tokenizer_model_name_or_path $speech_tokenizer \
        --flow_llm_config_path $PWD/conf/flow_config.json \
        --speaker_model_path  $speaker_model \
        --text_tokenizer_path $llm \
        --flow_model_path $mdir/model.safetensors \
        --data_path $data/test.syn.flow.jsonl \
        --inference_cfg_rate 0.7 \
        --save_dir $adir/mel_outputs \
        --n_timesteps 10
    mel_dir=$(realpath $adir)/mel_outputs
    audio_dir=$(realpath $adir)/audio_outputs
    mkdir -p $audio_dir
    pushd /jfs-hdfs/user/hao.yin/workspace/SLLM/hifi-gan
    python inference_e2e.py --input_mels_dir $mel_dir --output_dir $audio_dir --checkpoint_file pretrain_models/UNIVERSIAL_V1/g_02500000
    popd

    for x in `seq 0 19`; do echo $adir/audio_outputs/${x}_generated_e2e.wav; done > $adir/wav.list
    paste <(awk '{print $1}' data/test.syn.text) $adir/wav.list > $adir/wav.scp
    # Compute WER
    python tmp_tools/recognition.py $adir/wav.scp $adir/syn.text
    python tools/compute-wer.py --char=1 --v=1 \
        data/test.syn.text $adir/syn.text > $adir/syn.wer

    # Compute speaker similarity
    python tmp_tools/compute_similarity.py data/test.syn.jsonl $adir/wav.scp $adir/syn.sim
    # Overall performance
    tail $adir/syn.wer
    tail -n1 $adir/syn.sim
fi

