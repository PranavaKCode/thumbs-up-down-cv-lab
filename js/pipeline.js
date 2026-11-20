// Glue: photos -> silhouettes -> features -> model -> predictions.
import { preprocess } from './imgproc.js';
import { hogFeatures } from './hog.js';
import { fitScaler, standardize, trainLogisticRegression, probabilityUp } from './model.js';

export const LABELS = ['thumbs_up', 'thumbs_down'];

export async function buildDataset(photos, settings, onProgress) {
  // Every photo is processed with its OWN session's background reference.
  const features = [], labels = [], kept = [], rejected = [];
  for (let i = 0; i < photos.length; i++) {
    const photo = photos[i];
    const stages = preprocess(photo.image, photo.background.image, settings);
    if (stages.valid) {
      features.push(hogFeatures(stages.normalized));
      labels.push(photo.label);
      kept.push({ photo, normalized: stages.normalized });
    } else {
      rejected.push({ photo, normalized: stages.normalized, reason: stages.reason });
    }
    if (onProgress && (i % 8 === 7 || i === photos.length - 1)) await onProgress(i + 1, photos.length);
  }
  return { features, labels, kept, rejected };
}

export async function trainModel(dataset, settings, onStep) {
  const scaler = fitScaler(dataset.features);
  const X = dataset.features.map(f => standardize(scaler, f));
  const y = dataset.labels.map(label => (label === 'thumbs_up' ? 1 : -1));
  const { coef, intercept, history } = await trainLogisticRegression(X, y, { C: 1, onStep });
  return {
    format: 'thumbs-js-hog-v1',
    classNames: LABELS,
    scaler, coef, intercept, history,
    settings: { ...settings },                       // processing settings are frozen with the model
    camera: { sideFraction: 0.65, imageSize: 224 },
    trainingCounts: {
      thumbs_up: y.filter(v => v > 0).length,
      thumbs_down: y.filter(v => v < 0).length,
    },
    trainedAt: new Date().toISOString(),
  };
}

export function predict(model, features) {
  const up = probabilityUp(model, features);
  return { predicted: up > 0.5 ? 'thumbs_up' : 'thumbs_down', probabilityUp: up, score: Math.max(up, 1 - up) };
}

export async function evaluate(model, photos, onProgress) {
  const data = await buildDataset(photos, model.settings, onProgress);
  const results = data.kept.map((item, i) => ({ ...item, actual: item.photo.label, ...predict(model, data.features[i]) }));
  const matrix = { thumbs_up: { thumbs_up: 0, thumbs_down: 0 }, thumbs_down: { thumbs_up: 0, thumbs_down: 0 } };
  for (const r of results) matrix[r.actual][r.predicted]++;
  const correct = results.filter(r => r.predicted === r.actual).length;
  return { results, rejected: data.rejected, matrix, correct, usable: results.length, total: photos.length };
}
