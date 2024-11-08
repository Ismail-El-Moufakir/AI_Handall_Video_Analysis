import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as tf
import numpy as np
from PIL import Image
import sys

transform_model = tf.Compose([tf.ToTensor()])
model = fasterrcnn_resnet50_fpn(num_classes=5)
model.load_state_dict(torch.load("./handball_faster_rcnn_cpu.pth", map_location="cpu"))
model.eval()


colors = {
    1: (255, 0, 0),    # Class 1 - Red
    2: (0, 255, 0),    # Class 2 - Green
    3: (0, 0, 255),    # Class 3 - Blue
    4: (255, 255, 0),  # Class 4 - Cyan
    5: (255, 0, 255),  # Class 5 - Magenta
}


cap = cv2.VideoCapture("match.mp4")
if not cap.isOpened():
    print("Video opening error")
    exit()


fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter("output_match.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
processed_frames = 0

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frames += 1

    # Display progress in percentage
    progress = (processed_frames / total_frames) * 100
    sys.stdout.write(f"\rProcessing frame {processed_frames}/{total_frames} ({progress:.2f}%)")
    sys.stdout.flush()

    # Convert frame to PIL image and apply transform
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform_model(img_pil).unsqueeze(0)

    # Run model on the frame
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Draw bounding boxes and labels on the frame
    for i in range(len(predictions['boxes'])):
        box = predictions['boxes'][i].numpy().astype(int)
        score = predictions['scores'][i].item()
        label = int(predictions['labels'][i].item())

        if score > 0.5:
            color = colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            text = f"Class {label}: {score:.2f}"
            cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Write frame to output video
    output_video.write(frame)

    # Break loop with 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Finish the progress line
print("\nProcessing complete.")

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()
