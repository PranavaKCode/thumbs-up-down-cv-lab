// Image-processing steps for the lab.
// Every function works on plain arrays of numbers so you can read exactly
// what happens to each pixel. A "gray image" is { width, height, data }
// where data holds one brightness value (0 = black, 255 = white) per pixel.

export const DEFAULT_SETTINGS = {
  blurSize: 5,          // odd number: 1, 3, 5, 7, ...
  threshold: 25,        // how different a pixel must be from the background
  morphSize: 3,         // size of the clean-up kernel
  minForeground: 0.02,  // reject if the hand covers less than 2% of the square
  maxForeground: 0.75,  // reject if more than 75% of the square changed
  featureSize: 64,      // silhouettes are resized to 64 x 64
  padding: 4,
};

export function makeGray(width, height, data = new Uint8Array(width * height)) {
  return { width, height, data };
}

export function countWhite(img) {
  let count = 0;
  for (const value of img.data) if (value) count++;
  return count;
}

// Step 4: grayscale --------------------------------------------------------

export function toGray(photo) {
  // photo is an RGBA ImageData from a <canvas>: 4 numbers per pixel.
  const { width, height, data } = photo;
  const gray = makeGray(width, height);
  for (let i = 0; i < gray.data.length; i++) {
    const red = data[4 * i], green = data[4 * i + 1], blue = data[4 * i + 2];
    gray.data[i] = Math.round(0.299 * red + 0.587 * green + 0.114 * blue);
  }
  return gray;
}

// Step 2: background = median of several frames -----------------------------

export function medianImage(frames) {
  // For every pixel and color channel, sort the values from all frames
  // and keep the middle one. One odd frame (a flicker) can't move the median.
  const { width, height } = frames[0];
  const result = new ImageData(width, height);
  const values = new Array(frames.length);
  for (let i = 0; i < result.data.length; i++) {
    for (let f = 0; f < frames.length; f++) values[f] = frames[f].data[i];
    values.sort((a, b) => a - b);
    result.data[i] = values[values.length >> 1];
  }
  return result;
}

// Step 5A: blur and subtract ------------------------------------------------

export function mirrorIndex(i, n) {
  // Coordinates that fall off the edge are reflected back inside:
  // -1 becomes 1, and n becomes n - 2 (OpenCV's default border rule).
  if (n === 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

export function gaussianKernel(size) {
  // Bell-shaped weights that add up to 1. OpenCV picks sigma this way
  // when you give it only a kernel size.
  const sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8;
  const half = (size - 1) / 2;
  const weights = [];
  for (let i = 0; i < size; i++) weights.push(Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)));
  const total = weights.reduce((a, b) => a + b, 0);
  return weights.map(w => w / total);
}

export function gaussianBlur(img, size) {
  const { width: w, height: h, data } = img;
  if (size <= 1) return makeGray(w, h, data.slice());
  const k = gaussianKernel(size), r = (size - 1) / 2;
  // The 2-D blur is done as two 1-D passes: across each row, then down each column.
  const across = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let i = -r; i <= r; i++) sum += k[i + r] * data[y * w + mirrorIndex(x + i, w)];
      across[y * w + x] = sum;
    }
  const out = makeGray(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let i = -r; i <= r; i++) sum += k[i + r] * across[mirrorIndex(y + i, h) * w + x];
      out.data[y * w + x] = Math.round(sum);
    }
  return out;
}

export function absDiff(a, b) {
  // How much did each pixel change? Brighter or darker both count.
  const out = makeGray(a.width, a.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = Math.abs(a.data[i] - b.data[i]);
  return out;
}

// Step 5B: threshold -----------------------------------------------------------

export function threshold(img, limit) {
  // Changed more than the limit -> white (255). Otherwise -> black (0).
  const out = makeGray(img.width, img.height);
  for (let i = 0; i < out.data.length; i++) out.data[i] = img.data[i] > limit ? 255 : 0;
  return out;
}

// Step 5C: opening and closing -------------------------------------------------

export function ellipseKernel(size) {
  // A small round "stamp", listed as (dx, dy) offsets from its center.
  // Size 3 gives a plus sign; bigger sizes look like a disc.
  const r = Math.floor(size / 2), offsets = [];
  for (let dy = -r; dy <= r; dy++) {
    const reach = r === 0 ? 0 : Math.round(r * Math.sqrt(1 - (dy * dy) / (r * r)));
    for (let dx = -reach; dx <= reach; dx++) offsets.push([dx, dy]);
  }
  return offsets;
}

export function morph(img, offsets, operation) {
  // erode:  a pixel stays white only if EVERY pixel under the stamp is white.
  // dilate: a pixel becomes white if ANY pixel under the stamp is white.
  const { width: w, height: h, data } = img;
  const erode = operation === 'erode';
  const out = makeGray(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let value = erode ? 255 : 0;
      for (const [dx, dy] of offsets) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const v = data[yy * w + xx];
        value = erode ? Math.min(value, v) : Math.max(value, v);
      }
      out.data[y * w + x] = value;
    }
  return out;
}

export function openMask(mask, size) {
  // Opening = erode, then dilate: removes white specks smaller than the stamp.
  const kernel = ellipseKernel(size);
  return morph(morph(mask, kernel, 'erode'), kernel, 'dilate');
}

export function closeMask(mask, size) {
  // Closing = dilate, then erode: fills black gaps smaller than the stamp.
  const kernel = ellipseKernel(size);
  return morph(morph(mask, kernel, 'dilate'), kernel, 'erode');
}

// Step 6: largest region, crop, and resize ------------------------------------

export function largestComponent(mask) {
  // Flood-fill each connected white region (8 neighbours count as touching)
  // and keep only the biggest one.
  const { width: w, height: h, data } = mask;
  const region = new Int32Array(w * h);          // 0 = not visited yet
  const stack = new Int32Array(w * h);
  let regionCount = 0, best = 0, bestArea = 0;
  for (let start = 0; start < data.length; start++) {
    if (!data[start] || region[start]) continue;
    regionCount++;
    let area = 0, top = 0;
    stack[top++] = start;
    region[start] = regionCount;
    while (top > 0) {
      const p = stack[--top], x = p % w, y = (p - x) / w;
      area++;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          const q = yy * w + xx;
          if (data[q] && !region[q]) { region[q] = regionCount; stack[top++] = q; }
        }
    }
    if (area > bestArea) { bestArea = area; best = regionCount; }
  }
  const out = makeGray(w, h);
  for (let i = 0; i < out.data.length; i++) out.data[i] = best && region[i] === best ? 255 : 0;
  return out;
}

export function boundingBox(mask) {
  let minX = Infinity, minY = Infinity, maxX = -1, maxY = -1;
  for (let y = 0; y < mask.height; y++)
    for (let x = 0; x < mask.width; x++)
      if (mask.data[y * mask.width + x]) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
  if (maxX < 0) return null;
  return { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

export function normalizeSilhouette(mask, size = 64, padding = 4) {
  // Crop to the white shape, scale it to fit (keeping its proportions),
  // and center it. Never rotate or flip: thumb direction is the label!
  const out = makeGray(size, size);
  const box = boundingBox(mask);
  if (!box) return out;
  const scale = (size - 2 * padding) / Math.max(box.width, box.height);
  const newWidth = Math.max(1, Math.round(box.width * scale));
  const newHeight = Math.max(1, Math.round(box.height * scale));
  const left = Math.floor((size - newWidth) / 2), top = Math.floor((size - newHeight) / 2);
  for (let y = 0; y < newHeight; y++)
    for (let x = 0; x < newWidth; x++) {
      // "Nearest neighbour": copy the closest pixel from the cropped box.
      const sx = box.x + Math.min(box.width - 1, Math.floor((x * box.width) / newWidth));
      const sy = box.y + Math.min(box.height - 1, Math.floor((y * box.height) / newHeight));
      out.data[(top + y) * size + left + x] = mask.data[sy * mask.width + sx];
    }
  return out;
}

// All of the above, in order. Used for training, testing, and live video. -----

export function preprocess(photo, background, settings) {
  const gray = toGray(photo);
  const backgroundGray = toGray(background);
  const blurred = gaussianBlur(gray, settings.blurSize);
  const backgroundBlurred = gaussianBlur(backgroundGray, settings.blurSize);
  const difference = absDiff(blurred, backgroundBlurred);
  const binary = threshold(difference, settings.threshold);
  const opened = openMask(binary, settings.morphSize);
  const cleaned = closeMask(opened, settings.morphSize);
  const mask = largestComponent(cleaned);

  const foregroundFraction = countWhite(mask) / mask.data.length;
  const changedFraction = countWhite(cleaned) / cleaned.data.length;
  let valid = true, reason = 'Usable foreground';
  if (changedFraction > settings.maxForeground) {
    valid = false;
    reason = 'Too much changed: keep the camera still or record the background again';
  } else if (foregroundFraction < settings.minForeground) {
    valid = false;
    reason = 'No clear foreground: move your hand into the square';
  }
  const normalized = normalizeSilhouette(mask, settings.featureSize, settings.padding);
  return {
    gray, backgroundGray, blurred, backgroundBlurred, difference, binary, opened, cleaned,
    mask, box: boundingBox(mask), normalized, valid, reason, foregroundFraction, changedFraction,
  };
}
