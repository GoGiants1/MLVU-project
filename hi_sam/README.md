## 수정사항
- arg.parser에 model_name 추가(output file 구분하기 위함)
- input_image, output_image 디렉토리 추가(input file에 image 넣어두면 자동으로 output file에 text segmentation 추가)

## 명령어

- python run_hisam.py --checkpoint pretrained_checkpoint/sam_tss_l_hiertext.pth --model-type vit_l --input input_image --output output_image/ --model_name sam_tss
- sam_tss 모델 사용하는 명령어
- python run_hisam.py --checkpoint pretrained_checkpoint/hi_sam_l.pth --model-type vit_l --input input_image --output output_image/ --hier_det --model_name Hi_SAM
- hi-SAM 모델 사용하는 명령어

## TODO
- cuda GPU 할당 efficient하게 하기
- Hi-SAM model 쓸때 input point 어떻게 잡아줄 것인가......