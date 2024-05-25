

target_list=[]

target_images=[1,2,3,4]

while len(target_list)<5:

            for img in target_images:
                if len(target_list)==5:
                        break
                
                target_list.append(img)
print(target_list)