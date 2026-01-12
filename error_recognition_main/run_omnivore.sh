#!/bin/bash

SPLITS=('recordings' 'step' 'person' 'environment')
VARIANTS=('MLP' 'Transformer' 'RNN')
CKPT_DIRECTORY_PATH="/content/drive/MyDrive/PoliTo/Project/error_recognition-main/checkpoints"
BACKBONE=("omnivore")
FEATURES_DIRECTORY="/content/drive/MyDrive/PoliTo/Project/1s"
# "/data/rohith/captain_cook/features/gopro/segments_2"
TASK_NAME=("error_recognition" "error_category_recognition")
# Error categories for category recognition
ERROR_CATEGORIES=("TechniqueError" "PreparationError" "TemperatureError" "MeasurementError" "TimingError")

# Move to project root so python can find train_er.py
cd /content/drive/MyDrive/DAAI/error_recognition-main

# Function name corrected for typo and best practice
generate_run_scripts() {
    for split in "${SPLITS[@]}"; do
        for variant in "${VARIANTS[@]}"; do
            for task in "${TASK_NAME[@]}"; do

                if [[ "$task" == "error_recognition" ]]; then
                    echo "Training ER: split=$split variant=$variant"

                    if [[ "$variant" == "MLP" ]]; then
                        python train_er.py \
                            --task_name error_recognition \
                            --split $split \
                            --variant $variant \
                            --backbone $BACKBONE \
                            --ckpt_directory $CKPT_DIRECTORY_PATH

                    elif [[ "$variant" == "TRANSFORMER" ]]; then
                        python train_er.py \
                            --task_name error_recognition \
                            --split $split \
                            --variant $variant \
                            --backbone $BACKBONE \
                            --ckpt_directory $CKPT_DIRECTORY_PATH \
                            --lr 0.000001
                    else
                        python train_er.py \
                            --task_name error_recognition \
                            --split $split \
                            --variant $variant \
                            --backbone $BACKBONE \
                            --ckpt_directory $CKPT_DIRECTORY_PATH \
                            --lr 0.000001
                    fi
                fi

                if [[ "$task" == "error_category_recognition" ]]; then
                    for category in "${ERROR_CATEGORIES[@]}"; do
                        echo "Training ECR: split=$split variant=$variant category=$category"

                        if [[ "$variant" == "MLP" ]]; then
                            python train_ecr.py \
                                --task_name error_category_recognition \
                                --error_category $category \
                                --split $split \
                                --variant $variant \
                                --backbone $BACKBONE \
                                --ckpt_directory $CKPT_DIRECTORY_PATH

                        else
                            python train_ecr.py \
                                --task_name error_category_recognition \
                                --error_category $category \
                                --split $split \
                                --variant $variant \
                                --backbone $BACKBONE \
                                --ckpt_directory $CKPT_DIRECTORY_PATH \
                                --lr 0.000001
                        fi
                    done
                fi

            done
        done
    done
}

generate_run_scripts
