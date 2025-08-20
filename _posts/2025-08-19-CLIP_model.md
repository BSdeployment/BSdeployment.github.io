---
title: "Open ai Model - התאמה בין תמונה לטקסט"
date: 2025-08-19
categories: compute vision pytorch
---

# 🖼️ מהו CLIP ולמה הוא כל כך מדובר?

המודל **CLIP** (ראשי תיבות: *Contrastive Language–Image Pretraining*) פותח על ידי **OpenAI** והוא אחד המודלים הפופולריים ביותר בתחום ה־AI לשילוב בין תמונות וטקסט. הרעיון המרכזי הוא לאמן רשת נוירונים כך שתבין **קשרים סמנטיים** בין טקסט לתמונה.

במילים פשוטות: CLIP יודע למפות טקסטים ותמונות לאותו "מרחב משותף". שם ניתן למדוד את הקרבה ביניהם: תמונה של 🐱 תהיה קרובה מאוד ל"a photo of a cat" ורחוקה מאוד מ־"a photo of a car".

---

## 🎯 היעוד והמטרה של CLIP

* **חיפוש מבוסס טקסט** – מציאת תמונה מתוך טקסט (Text-to-Image Retrieval).
* **חיפוש מבוסס תמונה** – מציאת טקסט מתאים לתמונה (Image-to-Text Retrieval).
* **סיווג ללא אימון נוסף (Zero-Shot Classification)** – לתת למודל רשימת תוויות והוא יודע להתאים את התמונה בלי לאמן מחדש.

CLIP הפך לפופולרי מאוד בעולם ה־AI כי הוא פותח גשר חזק בין שני עולמות – טקסט ותמונה. היום הוא משמש גם כתשתית בהרבה מערכות מסחריות, מחקריות ואפילו במודלי יצירה מתקדמים (כגון DALL·E ו־Stable Diffusion).

---

## 💡 שימושים נפוצים

* 🖼️ **סיווג תמונות** לפי תוויות טקסטואליות בלי לאמן מודל ייעודי.
* 🔍 **חיפוש תמונה על בסיס טקסט** (למשל: "מצא לי את כל התמונות של כלבים בתיקייה").
* 🔄 **השוואת תמונה לתמונה** – חיפוש תמונות דומות.
* 🧩 **חיבור למודלים נוספים** (כמו captioning או יצירת תמונות).

---

## 📌 דוגמת קוד 1 – מציאת הטקסט שמתאים לתמונה (Image → Text)

```python
import torch
import clip
from PIL import Image

# טוען את המודל
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# תמונה
image = preprocess(Image.open("car.jpg")).unsqueeze(0).to(device)

# תוויות טקסטואליות
texts = ["a photo of a car", "a photo of a cat", "a photo of a plane"]
text_tokens = clip.tokenize(texts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # דמיון
    similarity = (image_features @ text_features.T).softmax(dim=-1)

# הצגת התוצאה
best_match = texts[similarity.argmax().item()]
print("התמונה מתאימה ביותר ל:", best_match)
```

📸➡️📝 כאן המודל מקבל תמונה ומחזיר איזו תווית טקסטואלית מתאימה לה.

---

## 📌 דוגמת קוד 2 – מציאת תמונה מתאימה לטקסט (Text → Image)

```python
import torch
import clip
from PIL import Image
import os

# טוען את המודל
model, preprocess = clip.load("ViT-B/32", device=device)

# תווית (טקסט)
query = "a photo of a ship"
text_features = model.encode_text(clip.tokenize([query]).to(device))

# תיקיית תמונות
folder = "images/"
images, image_features_list = [], []

for file in os.listdir(folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        img = Image.open(os.path.join(folder, file))
        images.append((file, img))
        with torch.no_grad():
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_tensor)
            image_features_list.append(img_feat)

# חישוב דמיון
similarities = [ (fname, (feat @ text_features.T).item()) for (fname,_), feat in zip(images, image_features_list) ]

# המיון מהגבוה לנמוך
similarities.sort(key=lambda x: x[1], reverse=True)

print("התמונה הדומה ביותר לטקסט:", similarities[0][0])
images[0][1].show()
```

📝➡️📸 כאן המודל מקבל טקסט (למשל "ship") ומחזיר איזו תמונה הכי מתאימה מתוך התיקייה.

---

## 🚀 סיכום

CLIP הוא מודל **פופולרי מאוד** בתעשייה בזכות הגמישות שלו – הוא יכול לשמש הן לסיווג תמונות, הן לחיפוש, והן כשכבת בסיס במודלים מורכבים יותר.

כוחו המרכזי: היכולת להבין תמונות וטקסטים באותו מרחב סמנטי, מה שמאפשר יישומים רבים בלמידת מכונה ובחיפוש חכם.
