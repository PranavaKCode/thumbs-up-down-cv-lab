// Histogram of Oriented Gradients (HOG): turns a 64 x 64 silhouette into
// 1,764 numbers that describe where its edges are and which way they run.
import { makeGray, mirrorIndex } from './imgproc.js';

export const HOG = { cellSize: 8, blockSize: 16, blockStride: 8, bins: 9 };

export function gradients(img) {
  // For each pixel: how fast does brightness change left->right (gx) and
  // top->bottom (gy)? The strength and angle of that change describe an edge.
  const { width: w, height: h, data } = img;
  const magnitude = new Float32Array(w * h), angle = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const gx = data[y * w + mirrorIndex(x + 1, w)] - data[y * w + mirrorIndex(x - 1, w)];
      const gy = data[mirrorIndex(y + 1, h) * w + x] - data[mirrorIndex(y - 1, h) * w + x];
      let degrees = (Math.atan2(gy, gx) * 180) / Math.PI;  // -180 .. 180
      if (degrees < 0) degrees += 180;                        // fold to 0 .. 180:
      if (degrees >= 180) degrees -= 180;                     // an edge has no "front"
      magnitude[y * w + x] = Math.hypot(gx, gy);
      angle[y * w + x] = degrees;
    }
  return { magnitude, angle };
}

function addToBins(hist, offset, bins, degrees, amount) {
  // Split the vote between the two nearest direction bins (20 degrees apart).
  const position = degrees / (180 / bins) - 0.5;
  const low = Math.floor(position), share = position - low;
  hist[offset + ((low + bins) % bins)] += amount * (1 - share);
  hist[offset + ((low + 1) % bins)] += amount * share;
}

export function hogFeatures(img) {
  const { cellSize, blockSize, blockStride, bins } = HOG;
  const w = img.width;
  const { magnitude, angle } = gradients(img);
  const blocksAcross = (w - blockSize) / blockStride + 1;   // 7 for a 64-pixel image
  const sigma = blockSize / 4;                              // pixels near a block's center count more
  const perBlock = 4 * bins;                                // 2 x 2 cells, 9 bins each
  const features = new Float32Array(blocksAcross * blocksAcross * perBlock);

  for (let by = 0; by < blocksAcross; by++)
    for (let bx = 0; bx < blocksAcross; bx++) {
      const hist = new Float32Array(perBlock);
      for (let py = 0; py < blockSize; py++)
        for (let px = 0; px < blockSize; px++) {
          const p = (by * blockStride + py) * w + (bx * blockStride + px);
          if (!magnitude[p]) continue;
          const dx = px - blockSize / 2 + 0.5, dy = py - blockSize / 2 + 0.5;
          const weight = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
          const cell = (py < cellSize ? 0 : 2) + (px < cellSize ? 0 : 1);
          addToBins(hist, cell * bins, bins, angle[p], magnitude[p] * weight);
        }
      // "L2-Hys" normalization: scale to length 1, cap each value at 0.2, scale again.
      // This makes the numbers about edge *shape*, not about how bright the edges are.
      let sum = 0;
      for (const v of hist) sum += v * v;
      let scale = 1 / (Math.sqrt(sum) + 0.1 * perBlock);
      sum = 0;
      for (let i = 0; i < perBlock; i++) {
        hist[i] = Math.min(hist[i] * scale, 0.2);
        sum += hist[i] * hist[i];
      }
      scale = 1 / (Math.sqrt(sum) + 1e-3);
      for (let i = 0; i < perBlock; i++) hist[i] *= scale;
      features.set(hist, (by * blocksAcross + bx) * perBlock);
    }
  return features;
}

// ---- Helpers for pictures (not used by the classifier) ----------------------

export function edgeStrength(img) {
  const { magnitude } = gradients(img);
  let max = 0;
  for (const v of magnitude) if (v > max) max = v;
  const out = makeGray(img.width, img.height);
  for (let i = 0; i < magnitude.length; i++) out.data[i] = max ? Math.round((255 * magnitude[i]) / max) : 0;
  return out;
}

export function cellHistograms(img) {
  const { cellSize, bins } = HOG;
  const cells = img.width / cellSize;
  const { magnitude, angle } = gradients(img);
  const values = new Float32Array(cells * cells * bins);
  for (let y = 0; y < img.height; y++)
    for (let x = 0; x < img.width; x++) {
      const p = y * img.width + x;
      if (!magnitude[p]) continue;
      const cell = Math.floor(y / cellSize) * cells + Math.floor(x / cellSize);
      addToBins(values, cell * bins, bins, angle[p], magnitude[p]);
    }
  return { cells, bins, values };
}

export function weightGlyphs(coef, featureSize = 64) {
  // Add up the model's weights for each (cell, direction), across the blocks
  // that contain that cell. Positive weights vote thumbs up, negative thumbs down.
  const { cellSize, blockSize, blockStride, bins } = HOG;
  const cells = featureSize / cellSize;
  const blocksAcross = (featureSize - blockSize) / blockStride + 1;
  const sum = new Float32Array(cells * cells * bins);
  let i = 0;
  for (let by = 0; by < blocksAcross; by++)
    for (let bx = 0; bx < blocksAcross; bx++)
      for (let c = 0; c < 4; c++) {
        const cy = by + (c >> 1), cx = bx + (c & 1);
        for (let b = 0; b < bins; b++) sum[(cy * cells + cx) * bins + b] += coef[i++];
      }
  return {
    up: { cells, bins, values: sum.map(v => Math.max(v, 0)) },
    down: { cells, bins, values: sum.map(v => Math.max(-v, 0)) },
  };
}

export function drawGlyphs(canvas, { cells, bins, values }, color, cellPixels = 32) {
  canvas.width = canvas.height = cells * cellPixels;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  for (let i = 1; i < cells; i++) {
    ctx.beginPath(); ctx.moveTo(i * cellPixels + 0.5, 0); ctx.lineTo(i * cellPixels + 0.5, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cellPixels + 0.5); ctx.lineTo(canvas.width, i * cellPixels + 0.5); ctx.stroke();
  }
  let max = 0;
  for (const v of values) if (v > max) max = v;
  if (!max) return;
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  for (let cy = 0; cy < cells; cy++)
    for (let cx = 0; cx < cells; cx++)
      for (let b = 0; b < bins; b++) {
        const v = values[(cy * cells + cx) * bins + b] / max;
        if (v < 0.04) continue;
        // Draw the edge itself, which runs at right angles to the gradient.
        const theta = (((b + 0.5) * 180) / bins + 90) * (Math.PI / 180);
        const len = v * cellPixels * 0.48;
        const x = (cx + 0.5) * cellPixels, y = (cy + 0.5) * cellPixels;
        ctx.globalAlpha = 0.25 + 0.75 * v;
        ctx.lineWidth = 1 + 2 * v;
        ctx.beginPath();
        ctx.moveTo(x - Math.cos(theta) * len, y - Math.sin(theta) * len);
        ctx.lineTo(x + Math.cos(theta) * len, y + Math.sin(theta) * len);
        ctx.stroke();
      }
  ctx.globalAlpha = 1;
}
