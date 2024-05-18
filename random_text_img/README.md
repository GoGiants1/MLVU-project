pretrained_checkpoint 안에 hi_sam_h.pth 넣으시오  

input text prompt: change “sleeper” to “bear”
위 Sleeper를  hi sam으로 따고 박스 마스킹을 한다. (이 때 하얀색 배경) (no text sttroke) 
그런다음 박스의 중앙점을 pillow image draw text로 전달해서 해당 좌표에 새로운 텍스트(”bear”)를 입력하면서 “bear” 텍스트가 해당좌표에 보이게끔 한다. (이 때 bear 텍스트는 검정색이고 하얀색 배경이다. 이런식으로 마스크 만든) 
이 때 bear 의 폰트는 위 sleeper의 폰트와 다르다. (그냥 pillow에서 제공하는 폰트를 따른다) 
그리고 bear 텍스트는 하얀색이면서 검정색 배경인 마스크 도 만든다. (Text stroke seg) 
Pillow 로 만든 scene text 에 굵기가 존재해야한다.