// Standardization and logistic regression, written out by hand.
// Same recipe as scikit-learn's make_pipeline(StandardScaler(),
// LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear")).

export function sigmoid(t) {
  // Squashes any score into a probability between 0 and 1.
  return t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t));
}

function softplus(t) {
  // log(1 + e^t), computed without overflowing for big t.
  return t > 0 ? t + Math.log1p(Math.exp(-t)) : Math.log1p(Math.exp(t));
}

function dot(a, b) {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += a[j] * b[j];
  return s;
}

export function fitScaler(rows) {
  // Learn each feature's average and spread from the TRAINING data only.
  const n = rows.length, d = rows[0].length;
  const mean = new Float64Array(d), scale = new Float64Array(d);
  for (const row of rows) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
  for (const row of rows) for (let j = 0; j < d; j++) scale[j] += (row[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) {
    const spread = Math.sqrt(scale[j]);
    scale[j] = spread > 1e-12 ? spread : 1;   // a feature that never changes is left alone
  }
  return { mean, scale };
}

export function standardize(scaler, features) {
  const out = new Float64Array(features.length);
  for (let j = 0; j < features.length; j++) out[j] = (features[j] - scaler.mean[j]) / scaler.scale[j];
  return out;
}

function choleskySolve(M, n, b) {
  // Solves M x = b for a symmetric positive-definite n x n matrix M.
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = M[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j * n + j];
    }
  const y = new Float64Array(n), x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}

export async function trainLogisticRegression(X, y, { C = 1, maxSteps = 30, tolerance = 1e-4, onStep } = {}) {
  // X: standardized feature rows. y: +1 for thumbs up, -1 for thumbs down.
  // Goal: find weights w that make  sigmoid(w . x)  close to 1 for thumbs up
  // and close to 0 for thumbs down, while keeping the weights small.
  const n = X.length, d = X[0].length + 1;                 // +1 = a constant "bias" input
  const A = X.map(row => { const a = new Float64Array(d); a.set(row); a[d - 1] = 1; return a; });
  const nUp = y.filter(v => v > 0).length;
  const classWeight = y.map(v => n / (2 * (v > 0 ? nUp : n - nUp)));  // "balanced" classes

  // Loss = 0.5 * |w|^2  +  C * sum_i classWeight_i * log(1 + exp(-y_i * score_i))
  const objective = (w, scores) => {
    let loss = 0;
    for (let i = 0; i < n; i++) loss += classWeight[i] * softplus(-y[i] * scores[i]);
    return 0.5 * dot(w, w) + C * loss;
  };
  const report = (step, scores) => {
    let loss = 0, correct = 0;
    for (let i = 0; i < n; i++) {
      loss += softplus(-y[i] * scores[i]) / n;              // average cross-entropy
      if ((scores[i] > 0 ? 1 : -1) === y[i]) correct++;
    }
    return { step, loss, accuracy: correct / n };
  };

  // Gram matrix K[i][j] = x_i . x_j lets every Newton step be solved with an
  // n x n system instead of a 1,765 x 1,765 one.
  const K = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) K[i * n + j] = K[j * n + i] = dot(A[i], A[j]);

  let w = new Float64Array(d);               // every weight starts at zero: the model knows nothing
  let scores = new Float64Array(n);          // scores[i] = w . x_i
  let f = objective(w, scores), firstGradient = 0;
  const history = [];

  for (let step = 0; ; step++) {
    // Gradient: which way should each weight move to lower the loss?
    const p = scores.map((s, i) => sigmoid(y[i] * s));
    const g = Float64Array.from(w);
    for (let i = 0; i < n; i++) {
      const r = C * classWeight[i] * (p[i] - 1) * y[i];
      if (r) for (let j = 0; j < d; j++) g[j] += r * A[i][j];
    }
    const gradientSize = Math.sqrt(dot(g, g));
    if (step === 0) firstGradient = gradientSize;
    const entry = report(step, scores);
    history.push(entry);
    if (onStep) await onStep(entry);
    if (gradientSize <= tolerance * firstGradient || step >= maxSteps) break;

    // Newton step: use the curvature of the loss to aim the step (Woodbury identity).
    const q = p.map((pi, i) => Math.sqrt(C * classWeight[i] * pi * (1 - pi)));
    const M = new Float64Array(n * n);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) M[i * n + j] = (i === j ? 1 : 0) + q[i] * K[i * n + j] * q[j];
    const Ag = A.map(a => dot(a, g));
    const t = choleskySolve(M, n, Ag.map((v, i) => q[i] * v)).map((v, i) => q[i] * v);
    const direction = Float64Array.from(g, v => -v);
    for (let i = 0; i < n; i++) if (t[i]) for (let j = 0; j < d; j++) direction[j] += t[i] * A[i][j];
    const scoreChange = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += K[i * n + k] * t[k];
      scoreChange[i] = s - Ag[i];
    }

    // Line search: halve the step until the loss actually goes down.
    const slope = dot(g, direction);
    let size = 1, nextW = w, nextScores = scores, nextF = f;
    for (let tries = 0; tries < 30; tries++, size /= 2) {
      nextW = w.map((v, j) => v + size * direction[j]);
      nextScores = scores.map((v, i) => v + size * scoreChange[i]);
      nextF = objective(nextW, nextScores);
      if (nextF <= f + 1e-4 * size * slope) break;
    }
    w = nextW; scores = nextScores; f = nextF;
  }
  return { coef: w.slice(0, d - 1), intercept: w[d - 1], history };
}

export function probabilityUp(model, features) {
  // Standardize with the training averages, take the weighted sum, squash.
  let score = model.intercept;
  for (let j = 0; j < features.length; j++)
    score += model.coef[j] * ((features[j] - model.scaler.mean[j]) / model.scaler.scale[j]);
  return sigmoid(score);
}
