---
title: "Text Classification - סיווג טקסט - ניתוח טקסט - בסיס"
date: 2025-08-20
categories: text  pytorch
---

# סיווג טקסט עם PyTorch – יסודות ודוגמה מעשית

## יסודות סיווג טקסט

סיווג טקסט הוא משימה מרכזית בתחום עיבוד שפה טבעית (NLP). המטרה היא לחזות את התווית (Label) של טקסט מסוים, למשל:

* האם ביקורת היא חיובית או שלילית?
* האם הודעה היא ספאם או לא ספאם?
* סיווג לפי נושאים (ספורט, טכנולוגיה, פוליטיקה וכו')

### שלבי עבודה בסיסיים:

1. **המרת הטקסט לייצוג מספרי** – כי מודלים של למידת מכונה עובדים עם מספרים, לא מילים. לדוגמה:

   * **Bag-of-Words (BoW)**: יוצרים מילון של כל המילים בטקסטים, ומייצגים כל משפט כוקטור 0/1 או ספירות לפי נוכחות מילים.
2. **חלוקה לסט אימון וסט בדיקה** – כדי לאמן את המודל ולבדוק את ביצועיו.
3. **בניית מודל למידת מכונה** – למשל Logistic Regression או רשת נוירונים פשוטה.
4. **אימון והערכה** – המודל לומד קשרים בין המילים לבין התוויות, ומסווג טקסטים חדשים.

## Bag-of-Words ודוגמא עם PyTorch

בגישה של Bag-of-Words:

* כל משפט מתורגם לווקטור בגודל המילון.
* לכל מילה במילון מוקצה תא בווקטור.
* אם המילה מופיעה במשפט → הערך בתא = 1 (או מספר הפעמים שהיא מופיעה). אם לא → 0.

**דוגמה תאורטית:**

```text
משפט: "I loved this movie, it was fantastic!"
מילון: ['acting', 'amazing', 'and', 'boring', 'enjoyed', 'ever', 'fantastic', 'film', 'great', 'hated', 'it', 'long', 'loved', 'movie', 'really', 'story', 'terrible', 'the', 'this', 'too', 'was', 'what', 'worst']
וקטור BoW: [0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0]
```

> הערכים 1 מראים אילו מילים מהמילון מופיעות במשפט.

### איך המודל לומד:

* המודל מקבל את הוקטור כקלט ואת התווית (חיובי/שלילי) כיעד.
* הוא לומד משקלים לכל מילה: מילים חיוביות יקבלו משקל חיובי, מילים שליליות משקל שלילי.
* בסוף, צירוף המשקלים של המילים במשפט קובע את הסיווג.

## דוגמת קוד מלאה – אימון רשת פשוטה על IMDB

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# 1. קריאת הקובץ
df = pd.read_csv("IMDB_Dataset.csv")
df['label'] = df['sentiment'].map({'positive':1, 'negative':0})

# 2. וקטוריזציה
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review']).toarray()
y = df['label'].values

# 3. חלוקה לאימון ובדיקה
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Dataset ב-PyTorch
class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = IMDBDataset(X_train, y_train)
val_dataset = IMDBDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# 5. מודל רשת פשוטה
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN(input_dim=5000, hidden_dim=128)

# 6. Loss ו-Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. אימון בסיסי
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for epoch in range(5):
    model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # בדיקת דיוק על סט validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    print(f"Epoch {epoch+1}, Validation Accuracy: {correct/total:.4f}")

# 8. סיווג משפט חדש
def predict_sentiment(model, sentence, vectorizer, device="cpu"):
    model.eval()
    X = vectorizer.transform([sentence]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
    return "positive" if prediction == 1 else "negative"

# דוגמאות
print(predict_sentiment(model, "I loved this movie, it was amazing!", vectorizer, device))
print(predict_sentiment(model, "This was the worst film I have ever seen.", vectorizer, device))
```

---

### סיכום

* הגישה שהצגנו היא **Bag-of-Words + רשת Fully Connected**.
* היא פשוטה, מהירה, ומספקת הבנה בסיסית של איך מודלים לומדים קשרים בין מילים לתוויות.
* לצורך למידת מודלים מתקדמים יותר (למשל מודלי שפה כמו BERT/GPT), משתמשים ב-Embeddings ומבנים שמכילים מידע על **סדר והקשר מילים**.

