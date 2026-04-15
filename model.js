/**
 * GlucoPredict — model.js
 *
 * Logistic regression inference.
 *
 * Input order (12 values, matching the training pipeline):
 *   [0]  Day1 · Breakfast Before
 *   [1]  Day1 · Breakfast After
 *   [2]  Day1 · Lunch Before
 *   [3]  Day1 · Lunch After
 *   [4]  Day1 · Dinner Before
 *   [5]  Day1 · Dinner After
 *   [6]  Day2 · Breakfast Before
 *   [7]  Day2 · Breakfast After
 *   [8]  Day2 · Lunch Before
 *   [9]  Day2 · Lunch After
 *   [10] Day2 · Dinner Before
 *   [11] Day2 · Dinner After
 *
 * Formula:
 *   z_i   = (x_i - mean_i) / std_i
 *   logit = bias + sum(weights_i * z_i)
 *   prob  = sigmoid(logit)
 */

/* ── Model parameters (embedded from .npy files) ── */

const MODEL = {
  bias: -1.4483660114682484,

  mean: [
    108.442, 108.138, 107.398, 106.428, 106.308, 106.306,
    107.836, 108.504, 109.720, 111.520, 112.278, 113.258
  ],

  std: [
    14.88780159, 16.76099509, 18.57922486, 20.24669890,
    21.46189965, 22.41250464, 23.35065533, 24.20061951,
    25.14719866, 26.96126111, 28.64522153, 30.69187900
  ],

  weights: [
     0.02902265, -0.01089704, -0.11656612,  0.00637091,
     0.03343125,  0.04270184,  0.08657605,  0.05081668,
     0.06608268,  0.19601873,  0.33454834,  0.43744894
  ]
};

/* ── Input field IDs (must match index order above) ── */

const INPUT_IDS = [
  'd1_bf_pre', 'd1_bf_post',
  'd1_lu_pre', 'd1_lu_post',
  'd1_di_pre', 'd1_di_post',
  'd2_bf_pre', 'd2_bf_post',
  'd2_lu_pre', 'd2_lu_post',
  'd2_di_pre', 'd2_di_post'
];

/* ── Math helpers ── */

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

function inferLogit(values) {
  let logit = MODEL.bias;
  for (let i = 0; i < 12; i++) {
    const z = (values[i] - MODEL.mean[i]) / MODEL.std[i];
    logit += MODEL.weights[i] * z;
  }
  return logit;
}

/* ── DOM helpers ── */

function el(id) {
  return document.getElementById(id);
}

function showError(msg) {
  const e = el('error-msg');
  e.textContent = msg;
  e.classList.add('show');
}

function hideError() {
  el('error-msg').classList.remove('show');
}

function animateBar(barId, widthPct, delay) {
  setTimeout(() => {
    el(barId).style.width = widthPct + '%';
  }, delay);
}

/* ── Main predict function (called by button) ── */

function predict() {
  /* 1. Collect & validate inputs */
  const values = [];
  let hasError = false;

  INPUT_IDS.forEach(id => {
    const input = el(id);
    input.classList.remove('err');
    const raw = input.value.trim();
    const val = parseFloat(raw);

    if (raw === '' || isNaN(val)) {
      input.classList.add('err');
      hasError = true;
    } else {
      values.push(val);
    }
  });

  if (hasError) {
    showError('Please fill in all 12 readings before predicting.');
    return;
  }

  hideError();

  /* 2. Run model inference */
  const logit = inferLogit(values);
  const prob  = sigmoid(logit);
  const spike = prob >= 0.5;

  const spikePct = parseFloat((prob * 100).toFixed(1));
  const safePct  = parseFloat((100 - spikePct).toFixed(1));

  /* 3. Update badge & title */
  const badge = el('result-badge');
  const title = el('result-title');

  if (spike) {
    badge.textContent  = 'Spike likely';
    badge.className    = 'result-badge badge-spike';
    title.textContent  = 'Your glucose may spike tomorrow.';
  } else {
    badge.textContent  = 'No spike expected';
    badge.className    = 'result-badge badge-safe';
    title.textContent  = 'Your glucose looks stable for tomorrow.';
  }

  /* 4. Update probability labels */
  el('prob-val').textContent = spikePct + '%';
  el('safe-val').textContent = safePct  + '%';

  /* 5. Update bar colors */
  const spikeFill = el('prob-fill');
  spikeFill.className = 'prob-bar-fill ' + (spike ? 'fill-spike' : 'fill-safe');
  spikeFill.style.width = '0%';
  el('safe-fill').style.width = '0%';

  /* 6. Show card, then animate bars */
  el('result-card').classList.add('show');
  animateBar('prob-fill', spikePct, 60);
  animateBar('safe-fill', safePct,  60);

  /* 7. Scroll result into view on mobile */
  el('result-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}