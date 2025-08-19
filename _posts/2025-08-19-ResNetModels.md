---
title: "Resnet Models - ראייה ממוחשבת - סיווג תמונות"
date: 2025-08-19
categories: compute vision pytorch
---

# 🏗️ מודלי ResNet – סקירה והסברים

מודלי **ResNet** (ראשי תיבות של *Residual Network*) נחשבים לאבן דרך משמעותית בעולם הראייה הממוחשבת והלמידה העמוקה. הם פותחו ב־2015 על ידי מיקרוסופט וזכו בתחרות **ImageNet** עם ביצועים פורצי דרך.

---

## 📌 מה החידוש של ResNet?

לפני ResNet, אחת הבעיות העיקריות בלמידה עמוקה הייתה שכשניסו להוסיף עוד שכבות לרשת נוירונים, **הביצועים לא השתפרו ואף הידרדרו** – תופעה שנקראת **Vanishing/Exploding Gradients**.

החידוש הגדול של ResNet:

* ✅ שימוש ב־**Residual Connections** – קפיצה (Skip Connection) שמדלגת על שכבות מסוימות ומאפשרת למודל "לעקוף" חישובים מיותרים.
* ✅ בכך נפתרה הבעיה של דעיכת גרדיאנטים, ואפשר היה לאמן רשתות עמוקות במיוחד (מאות שכבות).

---

## 🧩 הארכיטקטורה הבסיסית

האלמנט המרכזי של ResNet הוא **הבלוק השיורי (Residual Block)**:

```
Input → [Conv → BatchNorm → ReLU → Conv → BatchNorm] + Input → ReLU → Output
```

כלומר: הפלט של כמה שכבות מתווסף חזרה ל־Input. זה מאפשר למודל "ללמוד את השארית" (Residual), ולא את כל הייצוג מחדש.

---

## 🔢 גרסאות ResNet

* **ResNet-18, ResNet-34** – גרסאות קטנות יחסית, מתאימות למשימות פשוטות או משאבים מוגבלים.
* **ResNet-50** – גרסה פופולרית מאוד, משתמשת ב־Bottleneck Blocks.
* **ResNet-101, ResNet-152** – רשתות עמוקות מאוד, מתאימות למשימות מורכבות הדורשות דיוק גבוה.

---

## 🌍 שימושים נפוצים

מודלי ResNet נמצאים כמעט בכל תחום של ראייה ממוחשבת:

* 🖼️ סיווג תמונות (Image Classification).
* 🔲 זיהוי אובייקטים (Object Detection) – כבסיס למודלים כמו Faster R-CNN.
* ✂️ חלוקה (Segmentation) – בפרויקטים כמו Mask R-CNN.
* 🧬 רפואה – ניתוח תמונות רפואיות.
* 🚗 רכבים אוטונומיים – זיהוי סצנות מורכבות.

---

## 💡 החשיבות של ResNet

ResNet חולל מהפכה בכך שאפשר לראשונה לאמן רשתות **עמוקות מאוד** בקלות יחסית, והפך לבסיס לרבים מהמודלים המתקדמים יותר שהגיעו אחריו.

---

## 🧑‍💻 דוגמת קוד 1 – חיזוי עם ResNet-18

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# טעינת המודל מוכן מראש
model = models.resnet18(pretrained=True)
model.eval()

# טרנספורמציות לתמונה
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# טעינת תמונה
img = Image.open("image.jpg")
img_t = transform(img).unsqueeze(0)

# חיזוי
with torch.no_grad():
    outputs = model(img_t)
    _, predicted = outputs.max(1)

print(f"Predicted class index: {predicted.item()}")
```

---

## 🧑‍💻 דוגמת קוד 2 – אימון מחדש (Fine-Tuning) של ResNet

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# טרנספורמציות לדאטה חדש
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# טעינת מודל ResNet18
model = models.resnet18(pretrained=True)

# התאמת השכבה האחרונה למספר הקטגוריות החדשות (לדוגמה 2)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# אימון
for epoch in range(5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")
```

---

## 🚀 סיכום

מודלי ResNet הם מהחשובים ביותר בלמידה עמוקה וראייה ממוחשבת. הם שינו את הדרך בה ניתן לבנות רשתות נוירונים עמוקות ופתחו דלת למודלים מורכבים ומתקדמים עוד יותר. שליטה בהם היא צעד חשוב לכל מי שרוצה להתקדם בתחום.

