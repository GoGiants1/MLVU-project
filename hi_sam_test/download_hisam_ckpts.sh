#!/bin/bash

# Define the URL of the model checkpoint
SAM_H="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_L="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
SAM_B="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


Hisam_H="https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_h.pth"
Hisam_L="https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_l.pth"
Hisam_B="https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/hi_sam_b.pth"

detection_ctw1500="https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/line_detection_ctw1500.pth"
detection_totaltext="https://huggingface.co/GoGiants1/Hi-SAM/resolve/main/word_detection_totaltext.pth"

list=(SAM_H SAM_L SAM_B Hisam_H Hisam_L Hisam_B detection_ctw1500 detection_totaltext)

# Define the destination directory to save the checkpoint
destination_dir="pretrained_checkpoint"

# Create the destination directory if it doesn't exist
mkdir -p "$destination_dir"

cd "$destination_dir"

# Download the model checkpoints using wget and save them to the destination directory

for i in "${list[@]}"
do
    wget --content-disposition "${!i}"
done

echo "Model checkpoint downloaded successfully!"
