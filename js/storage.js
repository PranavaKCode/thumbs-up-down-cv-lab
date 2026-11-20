// Save the whole project as a ZIP (photos, backgrounds, silhouettes, model)
// and load it back later.
import { preprocess } from './imgproc.js';

const JSZIP_URL = 'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js';
const FORMAT = 'thumbs-lab-project-v1';
let jszip = null;

function loadJSZip() {
  if (window.JSZip) return Promise.resolve(window.JSZip);
  jszip ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = JSZIP_URL;
    script.onload = () => resolve(window.JSZip);
    script.onerror = () => {
      jszip = null;
      reject(new Error('Could not load the ZIP helper. Check your internet connection and try again.'));
    };
    document.head.append(script);
  });
  return jszip;
}

function toBlob(image) {
  // image: RGBA ImageData, or a gray image { width, height, data }.
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  let rgba = image;
  if (!(image instanceof ImageData)) {
    rgba = new ImageData(image.width, image.height);
    image.data.forEach((v, i) => { rgba.data.set([v, v, v, 255], 4 * i); });
  }
  canvas.getContext('2d').putImageData(rgba, 0, 0);
  return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}

async function fromBlob(blob) {
  const bitmap = await createImageBitmap(blob);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  return ctx.getImageData(0, 0, bitmap.width, bitmap.height);
}

function modelToJson(model) {
  if (!model) return null;
  const { history, ...rest } = model;
  return {
    ...rest,
    scaler: { mean: Array.from(model.scaler.mean), scale: Array.from(model.scaler.scale) },
    coef: Array.from(model.coef),
    history,
  };
}

function modelFromJson(json) {
  if (!json || json.format !== 'thumbs-js-hog-v1') return null;
  return {
    ...json,
    scaler: { mean: Float64Array.from(json.scaler.mean), scale: Float64Array.from(json.scaler.scale) },
    coef: Float64Array.from(json.coef),
  };
}

export async function saveProject(state, onProgress) {
  const JSZip = await loadJSZip();
  const zip = new JSZip();
  const root = zip.folder('thumbs_project');
  const backgrounds = new Map();
  for (const photo of state.photos) backgrounds.set(photo.background.id, photo.background);
  for (const bg of Object.values(state.backgrounds)) if (bg) backgrounds.set(bg.id, bg);

  const manifest = {
    format: FORMAT,
    savedAt: new Date().toISOString(),
    settings: state.settings,
    slots: Object.fromEntries(Object.entries(state.backgrounds).map(([k, bg]) => [k, bg ? bg.id : null])),
    backgrounds: [],
    photos: [],
    model: state.model ? 'thumbs_model.json' : null,
  };
  for (const bg of backgrounds.values()) {
    const file = `backgrounds/${bg.id}.png`;
    root.file(file, await toBlob(bg.image));
    manifest.backgrounds.push({ id: bg.id, file, frameSize: bg.frameSize, capturedAt: bg.capturedAt });
  }
  const processSettings = state.model ? state.model.settings : state.settings;
  for (let i = 0; i < state.photos.length; i++) {
    const photo = state.photos[i];
    const name = `${photo.split}/${photo.label}/${photo.id}.png`;
    root.file(`photos/${name}`, await toBlob(photo.image));
    const stages = preprocess(photo.image, photo.background.image, processSettings);
    if (stages.valid) root.file(`processed/${name}`, await toBlob(stages.normalized));
    manifest.photos.push({
      id: photo.id, split: photo.split, label: photo.label, file: `photos/${name}`,
      background: photo.background.id, capturedAt: photo.capturedAt,
    });
    if (onProgress && i % 10 === 9) await onProgress(i + 1, state.photos.length);
  }
  if (state.model) root.file('thumbs_model.json', JSON.stringify(modelToJson(state.model)));
  root.file('project.json', JSON.stringify(manifest, null, 2));
  return zip.generateAsync({ type: 'blob' });
}

export async function loadProject(file) {
  const JSZip = await loadJSZip();
  const zip = await JSZip.loadAsync(file);
  const manifestFile = zip.file(/(^|\/)project\.json$/)[0];
  if (!manifestFile) throw new Error('This ZIP has no project.json. Choose a ZIP saved by this lab.');
  const prefix = manifestFile.name.slice(0, -'project.json'.length);
  const manifest = JSON.parse(await manifestFile.async('string'));
  if (manifest.format !== FORMAT) throw new Error('This ZIP was not saved by this lab.');
  const read = async path => {
    const entry = zip.file(prefix + path);
    if (!entry) throw new Error('Missing file in ZIP: ' + path);
    return fromBlob(await entry.async('blob'));
  };
  const backgrounds = new Map();
  for (const bg of manifest.backgrounds) {
    backgrounds.set(bg.id, { id: bg.id, image: await read(bg.file), frameSize: bg.frameSize, capturedAt: bg.capturedAt });
  }
  const photos = [];
  for (const p of manifest.photos) {
    const background = backgrounds.get(p.background);
    if (!background) continue;
    photos.push({ id: p.id, split: p.split, label: p.label, capturedAt: p.capturedAt, image: await read(p.file), background });
  }
  let model = null;
  if (manifest.model) {
    const entry = zip.file(prefix + manifest.model);
    if (entry) model = modelFromJson(JSON.parse(await entry.async('string')));
  }
  const slots = {};
  for (const [key, id] of Object.entries(manifest.slots || {})) slots[key] = id ? backgrounds.get(id) ?? null : null;
  return { settings: manifest.settings, backgrounds: slots, photos, model };
}
