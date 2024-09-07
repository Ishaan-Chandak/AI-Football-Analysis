from ultralytics import YOLO

model = YOLO('models/best.pt')
model.to('cuda')

results = model.predict('inputs/08fd33_4.mp4', save=True, device="cuda:0")
print(results[0])

print("<=============================================>")

for box in results[0].boxes:
    print(box)