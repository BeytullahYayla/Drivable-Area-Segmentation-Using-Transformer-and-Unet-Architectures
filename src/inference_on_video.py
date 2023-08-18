import cv2
import numpy as np
import torch
from constant import *
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def get_predicted_images(model,image):
   
    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    pixel_values = feature_extractor_inference(image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    outputs = model(pixel_values=pixel_values)# logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
        # First, rescale logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(logits,
                    size=image.shape[:-1], # (height, width)
                    mode='bilinear',
                    align_corners=False)

    # Second, apply argmax on the class dimension
    seg = upsampled_logits.argmax(dim=1)[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3\
    for label, color in enumerate(RGB):
        color_seg[seg == label, :] = RGB[label]
    # Convert to BGR
    color_seg = color_seg[..., ::-1]

    # Show image + mask
    img = np.array(image)
    img[color_seg[:,:,1]==255,:]=(255,0,0)
    return img
  
        

def inference_from_video(model,video_path:str):
        # Video dosyasının yolu
   

    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    # Video'nun başarıyla açılıp açılmadığını kontrol et
    if not cap.isOpened():
        print("File can not opened")
        exit()
    frame_counter=0
    while True:
        # Tek bir kareyi yakala
        ret, frame = cap.read()

        # Video sona erdiyse çık
        if not ret:
            break

        # Karedaki işlemleri burada yapabilirsiniz
        prediction=get_predicted_images(model,frame)
        cv2.imshow("prediction",prediction)
        # Örneğin, kareyi görüntülemek için:
        cv2.imwrite(f"C:\\Users\\Beytullah\\Desktop\\outputs\\prediction{frame_counter}.jpg",prediction)
        

        # 'q' tuşuna basıldığında döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_counter+=1

    # Döngü bittiğinde işlemleri serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    model=torch.load("C:\\Users\\Beytullah\\Desktop\\models\\segformer.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_from_video(model,video_path="C:\\Users\\Beytullah\\Documents\\GitHub\\FordOtosan-L4Highway-Internship-Project\\data\\video\\output.avi")
