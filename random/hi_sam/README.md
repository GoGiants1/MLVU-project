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
- NMS 알고리즘 vs zero shot 최적화
- 이미지 크기 조절해서 테스트하기


## How to use Hi-SAM module
[text_segmentation.py](text_segmentation.py) 파일에서 여러 함수를 구현해두었습니다. 
1. `make_text_segmentation_args` 함수를 통해 argument를 설정합니다.
1. `load_auto_mask_generator` 함수를 통해 모델을 불러옵니다.
1. `run_text_detection` 함수를 통해 이미지를 입력하면 text detection을 수행합니다.
1. GPU에서 모델을 load or unload하고 싶다면 `unload_model` 혹은 `model_to_device` 함수를 사용합니다.