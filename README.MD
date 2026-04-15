# GlucoPredict

A lightweight, zero-dependency web app that predicts whether your blood glucose
will spike tomorrow based on the past two days of readings.

---

## Files

```
glucopredict/
├── index.html   — page structure & input form
├── style.css    — all styling (light + dark mode)
├── model.js     — model parameters & inference logic
└── README.md    — this file
```

---

## How to use

1. **Open `index.html` in any modern browser** — no server or internet connection
   required. Just double-click the file.

2. **Enter 12 glucose readings** (mg/dL):
   - Day 1 = two days ago (before/after breakfast, lunch, dinner)
   - Day 2 = yesterday (same 6 slots)

3. **Click "Predict tomorrow's glucose spike"** to see the result.

---

## How the model works

The model is a **logistic regression** trained on glucose time-series data.

**Inference steps:**

```
z_i   = (x_i − mean_i) / std_i          ← z-score normalisation
logit = bias + Σ (weights_i × z_i)       ← linear combination
prob  = sigmoid(logit) = 1 / (1 + e⁻ˡᵒᵍⁱᵗ)   ← spike probability
```

- If `prob ≥ 0.5` → spike predicted
- If `prob < 0.5` → no spike predicted

**Model parameters** (embedded in `model.js` from your `.npy` files):

| Parameter  | Shape  | Description                          |
|-----------|--------|--------------------------------------|
| `weights` | (12,)  | One weight per glucose reading       |
| `bias`    | scalar | Intercept term                       |
| `mean`    | (12,)  | Per-feature training mean (for z-score) |
| `std`     | (12,)  | Per-feature training std (for z-score)  |

**Input feature order** (must match training pipeline):

| Index | Day | Meal       | Timing |
|-------|-----|------------|--------|
| 0     | 1   | Breakfast  | Before |
| 1     | 1   | Breakfast  | After  |
| 2     | 1   | Lunch      | Before |
| 3     | 1   | Lunch      | After  |
| 4     | 1   | Dinner     | Before |
| 5     | 1   | Dinner     | After  |
| 6     | 2   | Breakfast  | Before |
| 7     | 2   | Breakfast  | After  |
| 8     | 2   | Lunch      | Before |
| 9     | 2   | Lunch      | After  |
| 10    | 2   | Dinner     | Before |
| 11    | 2   | Dinner     | After  |

---

## Updating the model parameters

If you retrain the model and get new `.npy` files, update the values in
`model.js` under the `MODEL` object:

```javascript
const MODEL = {
  bias: <new_value>,
  mean: [ /* 12 values */ ],
  std:  [ /* 12 values */ ],
  weights: [ /* 12 values */ ]
};
```

To extract values from `.npy` files in Python:

```python
import numpy as np

bias    = float(np.load('bias.npy'))
mean    = np.load('mean.npy').tolist()
std     = np.load('std.npy').tolist()
weights = np.load('weights.npy').tolist()

print('bias:', bias)
print('mean:', mean)
print('std:', std)
print('weights:', weights)
```

---

## Hosting online (optional)

Because this is plain HTML/CSS/JS with no backend, you can host it for free on:

- **GitHub Pages** — push the folder to a repo and enable Pages in Settings
- **Netlify** — drag & drop the folder at netlify.com/drop
- **Vercel** — `vercel deploy` from the folder

---

## Disclaimer

This tool is **not a medical device** and does not provide medical advice.
Predictions are probabilistic and may be inaccurate. Always consult a qualified
healthcare professional for clinical decisions.

---

## Requirements

- Any modern browser (Chrome, Firefox, Safari, Edge)
- No internet connection required
- No installation, no dependencies