# Copyright 2025 Binbin Zhang(binbzha@qq.com)

[ ! -s west ] && ln -s ../../west
[ ! -s tools ] && ln -s ../../tools
export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES="0"  # Change this to all your available gpus, such as "0,1,2,3"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')

llm=/bucket/output/jfs-hdfs/user/binbin.zhang/github/west/examples/aishell/tts/model/Qwen2.5-0.5B-Audio/

stage=train # data/train/decode
data=data
dir=exp/Qwen2.5-0.5B-Audio-pack6000-aidi-8gpus
steps=50000  # training steps


if [ $stage == "data" ] || [ $stage == "all" ]; then
    echo "Prepare required data"
fi


if [ $stage == "train" ] || [ $stage == "all" ]; then
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus west/bin/train.py \
        --model_type "codec_llm" \
        --extractor_type "tts_codec" \
        --llm_model_name_or_path $llm \
        --s3tokenizer_model_name_or_path $speech_tokenizer \
        --data_path $data/train.list \
        --bf16 True \
        --pack_size 1000 \
        --num_data_cycles 200 \
        --output_dir $dir \
        --max_steps $steps \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --save_strategy "steps" \
        --save_steps 1000 \
        --save_total_limit 100 \
        --learning_rate 3e-4 \
        --weight_decay 0.01 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
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
    mdir=$dir/checkpoint-8000
    # mdir=$dir/checkpoint-${steps}
    adir=$(echo $mdir | sed 's:exp:exp_audio:g')
    mkdir -p $adir
    adir=$(realpath $adir)
    python west/bin/decode.py \
        --model_type "codec_llm" \
        --extractor_type "tts_codec" \
        --codec_llm_model_path $mdir/model.safetensors \
        --s3tokenizer_model_name_or_path $speech_tokenizer \
        --data_path $data/test.syn.jsonl \
        --result_path $adir/result.txt
    paste <(awk '{print $1}' $data/test.syn.text) $adir/result.txt > $adir/result.hyp
    python tmp_tools/prepare_codec.py $adir/result.hyp data/test.syn.jsonl $adir/codec.jsonl

    # token2wav
    pushd /bucket/output/jfs-hdfs/user/binbin.zhang/gitlab/wenet/examples_tts/bigdata
    bash decode_tts_flow.sh $adir
    popd

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
