// Page wiring: cameras, buttons, pictures. The computer-vision and
// machine-learning code lives in imgproc.js, hog.js, model.js and pipeline.js.
import * as imgproc from './imgproc.js';
import * as hog from './hog.js';
import * as model from './model.js';
import * as pipeline from './pipeline.js';
import * as cam from './camera.js';
import { saveProject, loadProject } from './storage.js';

const { DEFAULT_SETTINGS, preprocess, countWhite, makeGray } = imgproc;
const { hogFeatures, cellHistograms, edgeStrength, weightGlyphs, drawGlyphs } = hog;
const { LABELS, buildDataset, trainModel, evaluate, predict } = pipeline;
const { camera, onCameraChange, startWebcam, startPracticeCamera, stopCamera, setPracticeGesture, snapshot, cameraReady } = cam;

const TARGET = { train: 60, test: 20 };
const NICE = { thumbs_up: 'thumbs up', thumbs_down: 'thumbs down' };
const SHORT = { thumbs_up: 'up', thumbs_down: 'down' };
const OTHER = { thumbs_up: 'thumbs_down', thumbs_down: 'thumbs_up' };
const BURST_MS = 400;
const NEXT_AFTER_BACKGROUND = {
  train: 'Next: collect training photos in step 3.',
  test: 'Next: collect test photos below.',
  live: 'Next: click Start predictions below.',
};

// ---- State and a tiny event bus -------------------------------------------

const state = {
  settings: { ...DEFAULT_SETTINGS },
  backgrounds: { train: null, test: null, live: null },
  photos: [],          // { id, split, label, image, background, capturedAt, thumb }
  exampleId: null,
  model: null,
  testReport: null,
  checked: false,
  liveFrames: 0,
  saved: false,
  unsaved: false,
};

const topics = new Map();
function on(topic, fn) {
  if (!topics.has(topic)) topics.set(topic, new Set());
  topics.get(topic).add(fn);
}
function emit(...names) {
  for (const name of names) for (const fn of topics.get(name) ?? []) fn();
}

// ---- Small helpers ----------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const pct = x => (100 * x).toFixed(1) + '%';
const fmt = n => n.toLocaleString('en-US');
let idCounter = 0;
const newId = prefix => `${prefix}_${Date.now().toString(36)}${(idCounter++).toString(36).padStart(3, '0')}`;
const photosOf = (split, label) => state.photos.filter(p => p.split === split && (!label || p.label === label));
const sameSize = (a, b) => a[0] === b[0] && a[1] === b[1];

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === false || value == null) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.filter(c => c != null && c !== false));
  return node;
}

function toImageData(img) {
  if (img instanceof ImageData) return img;
  const out = new ImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i++) {
    const v = img.data[i], p = 4 * i;
    out.data[p] = out.data[p + 1] = out.data[p + 2] = v;
    out.data[p + 3] = 255;
  }
  return out;
}

function paint(canvas, img) {
  const data = toImageData(img);
  if (canvas.width !== data.width) canvas.width = data.width;
  if (canvas.height !== data.height) canvas.height = data.height;
  canvas.getContext('2d').putImageData(data, 0, 0);
}

function blank(canvas) {
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}

const scratch = document.createElement('canvas');
function imageUrl(img, size) {
  paint(scratch, img);
  if (!size || size === scratch.width) return scratch.toDataURL('image/png');
  const small = document.createElement('canvas');
  small.width = small.height = size;
  const ctx = small.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(scratch, 0, 0, size, size);
  return small.toDataURL('image/jpeg', 0.85);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setStatus(node, ...parts) {
  node.replaceChildren(...parts);
}

// ---- Show the code ---------------------------------------------------------

const CODE = { ...imgproc, ...hog, ...model, ...pipeline, snapshot: cam.snapshot };

function highlight(source) {
  const escape = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c]);
  const tokens = /(\/\/[^\n]*)|('(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*")|\b(function|const|let|return|for|if|else|while|async|await|new|of|continue|break|throw|true|false|null)\b|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b/g;
  let html = '', last = 0;
  for (const m of source.matchAll(tokens)) {
    html += escape(source.slice(last, m.index));
    const kind = m[1] ? 'c' : m[2] ? 's' : m[3] ? 'k' : 'n';
    html += `<span class="tok-${kind}">${escape(m[0])}</span>`;
    last = m.index + m[0].length;
  }
  return html + escape(source.slice(last));
}

for (const pre of $$('pre[data-code]')) {
  const source = pre.dataset.code.split(',')
    .map(name => (CODE[name] ? CODE[name].toString() : `// ${name}: not found`))
    .join('\n\n');
  pre.innerHTML = highlight(source);
}

// ---- Camera views ------------------------------------------------------------

function createCameraView(host) {
  host.classList.add('cam');
  host.innerHTML = `
    <div class="cam-frame">
      <video muted playsinline autoplay></video>
      <div class="roi" hidden></div>
      <div class="cam-off">
        <p class="cam-off-title">Camera is off</p>
        <div class="cam-off-buttons">
          <button class="btn primary" data-act="webcam">Start camera</button>
          <button class="btn" data-act="practice">Use practice camera</button>
        </div>
        <p class="cam-error" role="alert"></p>
      </div>
    </div>`;
  const frame = $('.cam-frame', host), video = $('video', host), roi = $('.roi', host);
  const off = $('.cam-off', host), title = $('.cam-off-title', host), error = $('.cam-error', host);
  const buttons = $$('.cam-off button', host);
  $('[data-act=webcam]', host).addEventListener('click', () => startWebcam());
  $('[data-act=practice]', host).addEventListener('click', () => startPracticeCamera());
  let visible = false;

  const layout = () => {
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h || !camera.stream) return;
    frame.style.aspectRatio = `${w} / ${h}`;
    const side = Math.floor(Math.min(w, h) * cam.ROI_FRACTION);
    Object.assign(roi.style, {
      width: (side / w) * 100 + '%', height: (side / h) * 100 + '%',
      left: ((w - side) / 2 / w) * 100 + '%', top: ((h - side) / 2 / h) * 100 + '%',
    });
    roi.hidden = false;
  };
  video.addEventListener('loadedmetadata', layout);
  video.addEventListener('resize', layout);

  onCameraChange(c => {
    if (video.srcObject !== c.stream) video.srcObject = c.stream;
    off.hidden = Boolean(c.stream);
    if (!c.stream) roi.hidden = true;
    else layout();
    title.textContent = c.kind === 'starting' ? 'Waiting for camera permission...' : 'Camera is off';
    buttons.forEach(b => { b.disabled = c.kind === 'starting'; });
    error.textContent = c.error || '';
    if (c.stream && visible) video.play().catch(() => {});
  });
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (!camera.stream) return;
    if (visible) video.play().catch(() => {});
    else video.pause();
  }).observe(frame);
}

for (const host of $$('[data-camera]')) createCameraView(host);

// Top bar.
onCameraChange(c => {
  const dot = $('#camDot');
  dot.className = 'cam-dot ' + (c.stream ? 'on' : c.kind === 'starting' ? 'starting' : '');
  $('#camLabel').textContent =
    c.kind === 'webcam' ? 'webcam on' : c.kind === 'practice' ? 'practice camera:' :
    c.kind === 'starting' ? 'starting camera...' : 'camera off';
  const select = $('#camSelect');
  select.hidden = !(c.kind === 'webcam' && c.devices.length > 1);
  if (!select.hidden) {
    select.replaceChildren(...c.devices.map((d, i) =>
      el('option', { value: d.deviceId, text: d.label || `Camera ${i + 1}`, selected: d.deviceId === c.deviceId })));
  }
  $('#practiceBar').hidden = c.kind !== 'practice';
  for (const b of $$('#practiceBar button')) b.setAttribute('aria-pressed', String(b.dataset.gesture === c.gesture));
  $('#camStop').hidden = !c.stream;
});
$('#camSelect').addEventListener('change', e => startWebcam(e.target.value));
$('#camStop').addEventListener('click', () => stopCamera());
for (const b of $$('#practiceBar button')) b.addEventListener('click', () => setPracticeGesture(b.dataset.gesture));

// ---- Step 1: camera check ---------------------------------------------------------

$('#checkCapture').addEventListener('click', () => {
  const status = $('#checkStatus');
  try {
    const shot = snapshot();
    paint($('#checkCanvas'), shot.image);
    const [h, w] = shot.frameSize;
    $('#checkReadout').textContent =
      `Image received: 224 × 224 pixels × 3 color channels = ${fmt(224 * 224 * 3)} numbers. ` +
      `(Full camera frame: ${w} × ${h}.)`;
    status.textContent = 'Camera works. Next: step 2, record the empty background.';
    state.checked = true;
    emit('progress');
  } catch (error) {
    status.textContent = error.message;
  }
});

// ---- Background recorders (steps 2, 9, 10) --------------------------------------

function createBackgroundRecorder(host, key) {
  host.classList.add('cam-row');
  const camHost = el('div');
  const button = el('button', { class: 'btn primary', text: 'Capture background' });
  const status = el('p', { class: 'status', role: 'status' });
  const frames = Array.from({ length: 7 }, () => el('canvas', { width: 224, height: 224 }));
  const result = el('canvas', { width: 224, height: 224 });
  host.append(
    el('div', { class: 'cam-col' }, camHost, el('div', { class: 'controls' }, button), status),
    el('div', { class: 'side-col' },
      el('p', { class: 'mini-label', text: 'seven frames' }),
      el('div', { class: 'frames' }, ...frames),
      el('figure', { class: 'view' }, result, el('figcaption', { text: 'median of the seven = background reference' }))),
  );
  createCameraView(camHost);

  let busy = false;
  button.addEventListener('click', async () => {
    if (busy) return;
    if (!cameraReady()) { status.textContent = 'Start the camera first.'; return; }
    busy = true;
    button.disabled = true;
    try {
      if (camera.kind === 'practice' && camera.gesture !== 'none') {
        setPracticeGesture('none');
        await sleep(250);
      }
      frames.forEach(blank);
      const images = [];
      let frameSize = null;
      for (let i = 0; i < frames.length; i++) {
        status.textContent = `Recording frame ${i + 1} of 7. Keep your hand out of the square.`;
        const shot = snapshot();
        if (frameSize && !sameSize(frameSize, shot.frameSize)) throw new Error('The camera size changed while recording. Try again.');
        frameSize = shot.frameSize;
        images.push(shot.image);
        paint(frames[i], shot.image);
        await sleep(100);
      }
      status.textContent = 'Taking the median of each pixel...';
      await sleep(0);
      state.backgrounds[key] = { id: newId('bg'), image: imgproc.medianImage(images), frameSize, capturedAt: new Date().toISOString() };
      state.unsaved = true;
      status.textContent = 'Background recorded. ' + NEXT_AFTER_BACKGROUND[key];
      emit('backgrounds');
    } catch (error) {
      status.textContent = error.message;
    } finally {
      busy = false;
      button.disabled = false;
    }
  });

  on('backgrounds', () => {
    const bg = state.backgrounds[key];
    if (bg) paint(result, bg.image);
    else { blank(result); frames.forEach(blank); }
  });
}

for (const host of $$('[data-background]')) createBackgroundRecorder(host, host.dataset.background);

// ---- Photo collectors (steps 3 and 9) ----------------------------------------------

function createCollector(host, split) {
  const target = TARGET[split];
  host.innerHTML = `
    <div class="cam-row">
      <div class="cam-col">
        <div data-cam></div>
        <div class="seg label-toggle" role="group" aria-label="Gesture you are showing">
          <button data-label="thumbs_up" aria-pressed="true">label: thumbs up</button>
          <button data-label="thumbs_down" aria-pressed="false">label: thumbs down</button>
        </div>
        <div class="controls">
          <button class="btn primary" data-act="one">Capture one</button>
          <button class="btn" data-act="burst">Start burst</button>
        </div>
        <p class="status" role="status"></p>
      </div>
      <div class="side-col">
        <p class="mini-label">live mask preview (current settings)</p>
        <div class="mini-views">
          <figure class="view"><canvas data-preview="mask" width="224" height="224"></canvas><figcaption>mask</figcaption></figure>
          <figure class="view"><canvas data-preview="input" class="px" width="64" height="64"></canvas><figcaption>classifier input</figcaption></figure>
        </div>
        <p class="preview-note"></p>
      </div>
    </div>
    <div class="galleries">
      ${LABELS.map(label => `
        <div class="gallery" data-gallery="${label}">
          <div class="gallery-head">
            <span class="gallery-title">${NICE[label]}</span>
            <span class="count"></span>
            <button class="link" data-clear="${label}">Clear</button>
          </div>
          <div class="progress"><div></div></div>
          <div class="thumbs"></div>
        </div>`).join('')}
    </div>`;
  createCameraView($('[data-cam]', host));
  const status = $('.status', host);
  const burstButton = $('[data-act=burst]', host);
  const note = $('.preview-note', host);
  const maskCanvas = $('[data-preview=mask]', host), inputCanvas = $('[data-preview=input]', host);
  let label = 'thumbs_up', bursting = false, burstTimer = null, visible = false;
  const thumbs = new Map();

  function chooseLabel(next) {
    label = next;
    setBurst(false);
    for (const b of $$('[data-label]', host)) b.setAttribute('aria-pressed', String(b.dataset.label === label));
    if (camera.kind === 'practice') setPracticeGesture(label);
  }
  for (const b of $$('[data-label]', host)) b.addEventListener('click', () => chooseLabel(b.dataset.label));

  function captureOne() {
    const bg = state.backgrounds[split];
    if (!cameraReady()) throw new Error('Start the camera first.');
    if (!bg) throw new Error(split === 'train' ? 'Record the empty background in step 2 first.' : 'Record the test background above first.');
    const shot = snapshot();
    if (!sameSize(shot.frameSize, bg.frameSize)) throw new Error('The camera changed since the background was recorded. Record the background again.');
    const photo = { id: newId('photo'), split, label, image: shot.image, background: bg, capturedAt: new Date().toISOString() };
    photo.thumb = imageUrl(photo.image, 80);
    state.photos.push(photo);
    state.unsaved = true;
    emit('photos');
    const n = photosOf(split, label).length;
    const other = OTHER[label];
    status.textContent = n < target
      ? `Saved ${n} / ${target} ${NICE[label]} photos. Keep varying the gesture.`
      : `Saved ${n} / ${target} ${NICE[label]} photos. Target reached.` +
        (photosOf(split, other).length < target ? ` Now switch to ${NICE[other]}.` : '');
    return n;
  }

  function setBurst(onOff) {
    bursting = onOff;
    burstButton.textContent = onOff ? 'Pause burst' : 'Start burst';
    burstButton.classList.toggle('active', onOff);
    clearTimeout(burstTimer);
    if (onOff) burstTick();
  }
  function burstTick() {
    if (!bursting) return;
    try {
      if (captureOne() >= target) { setBurst(false); return; }
    } catch (error) {
      status.textContent = error.message;
      setBurst(false);
      return;
    }
    burstTimer = setTimeout(burstTick, BURST_MS);
  }

  $('[data-act=one]', host).addEventListener('click', () => {
    try { captureOne(); } catch (error) { status.textContent = error.message; }
  });
  burstButton.addEventListener('click', () => {
    if (bursting) { setBurst(false); status.textContent = 'Burst paused.'; return; }
    if (photosOf(split, label).length >= target) {
      status.textContent = `You already have ${target} ${NICE[label]} photos. Delete some (or Clear) to retake them, or use Capture one to add extras.`;
      return;
    }
    setBurst(true);
  });

  function removePhoto(photo) {
    const index = state.photos.indexOf(photo);
    if (index < 0) return;
    state.photos.splice(index, 1);
    state.unsaved = true;
    emit('photos');
    setStatus(status, `Deleted one ${NICE[photo.label]} photo. `, el('button', {
      class: 'link', text: 'Undo',
      onclick: () => {
        state.photos.splice(Math.min(index, state.photos.length), 0, photo);
        emit('photos');
        status.textContent = 'Photo restored.';
      },
    }));
  }

  for (const b of $$('[data-clear]', host)) {
    b.addEventListener('click', () => {
      const which = b.dataset.clear;
      const n = photosOf(split, which).length;
      if (!n || !confirm(`Delete all ${n} ${NICE[which]} ${split === 'train' ? 'training' : 'test'} photos?`)) return;
      setBurst(false);
      state.photos = state.photos.filter(p => !(p.split === split && p.label === which));
      emit('photos');
      status.textContent = `Deleted ${n} photos.`;
    });
  }

  function renderGalleries() {
    const alive = new Set(state.photos.map(p => p.id));
    for (const [id, node] of thumbs) if (!alive.has(id)) { node.remove(); thumbs.delete(id); }
    for (const which of LABELS) {
      const list = photosOf(split, which);
      const gallery = $(`[data-gallery=${which}]`, host);
      const box = $('.thumbs', gallery);
      list.forEach((photo, i) => {
        let node = thumbs.get(photo.id);
        if (!node) {
          node = el('button', { class: 'thumb', title: 'Click to delete this photo', onclick: () => removePhoto(photo) },
            el('img', { src: photo.thumb, alt: `${NICE[which]} photo` }));
          thumbs.set(photo.id, node);
        }
        if (box.children[i] !== node) box.insertBefore(node, box.children[i] ?? null);
      });
      $('.count', gallery).textContent = `${list.length} / ${target}`;
      $('.progress div', gallery).style.width = Math.min(100, (100 * list.length) / target) + '%';
      gallery.classList.toggle('complete', list.length >= target);
    }
  }
  on('photos', renderGalleries);
  renderGalleries();

  // Keep the practice camera's hand in sync with the chosen label.
  onCameraChange(c => {
    if (!c.stream) setBurst(false);
  });

  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(host);
  (async function previewLoop() {
    for (;;) {
      await sleep(180);
      if (!visible || !cameraReady()) continue;
      const bg = state.backgrounds[split];
      if (!bg) {
        note.textContent = split === 'train' ? 'Record the background (step 2) to see the mask preview.' : 'Record the test background to see the mask preview.';
        note.className = 'preview-note';
        blank(maskCanvas); blank(inputCanvas);
        continue;
      }
      try {
        const shot = snapshot();
        if (!sameSize(shot.frameSize, bg.frameSize)) {
          note.textContent = 'The camera changed. Record the background again.';
          note.className = 'preview-note bad';
          continue;
        }
        const result = preprocess(shot.image, bg.image, state.settings);
        paint(maskCanvas, result.mask);
        paint(inputCanvas, result.normalized);
        note.textContent = result.valid ? 'usable mask' : result.reason;
        note.className = 'preview-note ' + (result.valid ? 'good' : 'bad');
      } catch { /* camera stopped mid-frame */ }
    }
  })();
}

for (const host of $$('[data-collector]')) createCollector(host, host.dataset.collector);

// ---- Steps 4–7: follow one photo through the pipeline --------------------------

function renderPicker() {
  const picker = $('#examplePicker');
  const train = photosOf('train');
  if (!train.some(p => p.id === state.exampleId)) state.exampleId = train[0]?.id ?? null;
  picker.replaceChildren(...train.map(photo => el('button', {
    class: 'thumb' + (photo.id === state.exampleId ? ' selected' : ''),
    title: NICE[photo.label],
    'aria-pressed': String(photo.id === state.exampleId),
    onclick: () => { state.exampleId = photo.id; renderPicker(); queueExplorer(); },
  }, el('img', { src: photo.thumb, alt: NICE[photo.label] }), el('span', { class: 'tag', text: SHORT[photo.label] }))));
  $('[data-empty-note]').hidden = train.length > 0;
}

function paintBoxed(canvas, image, box) {
  paint(canvas, image);
  if (!box) return;
  const ctx = canvas.getContext('2d');
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#000000';
  ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.width - 1, box.height - 1);
  ctx.lineWidth = 1;
  ctx.strokeStyle = '#ffffff';
  ctx.strokeRect(box.x + 0.5, box.y + 0.5, box.width - 1, box.height - 1);
}

function paintVector(canvas, features) {
  // 1,764 numbers = a 42 x 42 grid of squares, brighter = bigger.
  const side = Math.ceil(Math.sqrt(features.length));
  const out = new ImageData(side, side);
  let max = 0;
  for (const v of features) if (v > max) max = v;
  for (let i = 0; i < side * side; i++) {
    const v = i < features.length && max ? features[i] / max : 0;
    const g = Math.round(255 * v);
    out.data.set([g, g, g, 255], 4 * i);
  }
  paint(canvas, out);
}

function drawHistogram(canvas, difference, limit) {
  const counts = new Array(256).fill(0);
  for (const v of difference.data) counts[v]++;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, L = 36, R = 10, T = 10, B = 26;
  const plotW = W - L - R, plotH = H - T - B, barW = plotW / 256;
  const maxLog = Math.log10(Math.max(...counts) + 1) || 1;
  const muted = cssVar('--muted'), text = cssVar('--text'), line = cssVar('--hair');
  ctx.clearRect(0, 0, W, H);
  ctx.font = '12px ' + cssVar('--font');
  ctx.fillStyle = muted;
  for (const tick of [0, 64, 128, 192, 255]) {
    const x = L + (tick + 0.5) * barW;
    ctx.textAlign = 'center';
    ctx.fillText(String(tick), x, H - 8);
  }
  ctx.strokeStyle = line;
  ctx.beginPath(); ctx.moveTo(L, T + plotH + 0.5); ctx.lineTo(W - R, T + plotH + 0.5); ctx.stroke();
  for (let v = 0; v < 256; v++) {
    if (!counts[v]) continue;
    const h = (Math.log10(counts[v] + 1) / maxLog) * plotH;
    ctx.fillStyle = v > limit ? text : muted;
    ctx.globalAlpha = v > limit ? 1 : 0.5;
    ctx.fillRect(L + v * barW, T + plotH - h, Math.max(1, barW - 0.5), h);
  }
  ctx.globalAlpha = 1;
  const x = L + (limit + 1) * barW;
  ctx.strokeStyle = text;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, T + plotH); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = text;
  ctx.textAlign = 'left';
  ctx.fillText(`threshold ${limit} → white`, Math.min(x + 6, W - 130), T + 12);
  ctx.save();
  ctx.translate(12, T + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillStyle = muted;
  ctx.fillText('pixels', 0, 0);
  ctx.restore();
}

let explorer = null;     // the stages of the photo being shown, for the pixel inspector
let explorerQueued = false;
function queueExplorer() {
  if (explorerQueued) return;
  explorerQueued = true;
  requestAnimationFrame(() => { explorerQueued = false; renderExplorer(); });
}

function renderExplorer() {
  const photo = state.photos.find(p => p.id === state.exampleId);
  const views = $$('[data-view]');
  if (!photo) {
    explorer = null;
    views.forEach(blank);
    $('#histogram').getContext('2d').clearRect(0, 0, 9999, 9999);
    for (const id of ['#binaryReadout', '#silhouetteReadout', '#featureReadout']) $(id).textContent = '';
    return;
  }
  const s = preprocess(photo.image, photo.background.image, state.settings);
  const features = hogFeatures(s.normalized);
  explorer = { photo, stages: s };
  const images = {
    original: photo.image, gray: s.gray, backgroundGray: s.backgroundGray,
    backgroundBlurred: s.backgroundBlurred, blurred: s.blurred, difference: s.difference,
    binary: s.binary, opened: s.opened, cleaned: s.cleaned, mask: s.mask,
    normalized: s.normalized, edges: edgeStrength(s.normalized),
  };
  const cells = cellHistograms(s.normalized);
  for (const canvas of views) {
    const name = canvas.dataset.view;
    if (name === 'boxed') paintBoxed(canvas, photo.image, s.box);
    else if (name === 'hog') drawGlyphs(canvas, cells, '#ffffff', 28);
    else if (name === 'vector') paintVector(canvas, features);
    else if (images[name]) paint(canvas, images[name]);
  }
  drawHistogram($('#histogram'), s.difference, state.settings.threshold);

  const values = [...new Set(s.binary.data)].sort((a, b) => a - b);
  $('#binaryReadout').textContent =
    `Mask pixel values: ${values.join(' and ')} · ${pct(countWhite(s.binary) / s.binary.data.length)} of pixels are white`;

  $('#silhouetteReadout').replaceChildren(
    el('span', { class: 'badge', text: (s.valid ? '[ok] ' : '[rejected] ') + s.reason + '.' }),
    ` Hand region: ${pct(s.foregroundFraction)} of the square · changed pixels: ${pct(s.changedFraction)}`,
  );
  $('#featureReadout').textContent =
    `Feature vector length: ${fmt(features.length)} = 7 × 7 blocks × 4 cells × 9 directions. ` +
    `First twelve values: ${Array.from(features.slice(0, 12), v => v.toFixed(3)).join(', ')}`;
}

// Pixel inspector for step 4.
for (const canvas of $$('[data-view=original], [data-view=gray]')) {
  canvas.addEventListener('pointermove', event => {
    if (!explorer) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(223, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * 224)));
    const y = Math.min(223, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * 224)));
    const i = y * 224 + x, d = explorer.photo.image.data;
    $('#pixelReadout').textContent =
      `Pixel (${x}, ${y}): red ${d[4 * i]}, green ${d[4 * i + 1]}, blue ${d[4 * i + 2]} → gray ${explorer.stages.gray.data[i]}`;
  });
}

// Sliders.
function syncSliders() {
  for (const input of $$('[data-setting]')) {
    input.value = state.settings[input.dataset.setting];
    input.nextElementSibling.textContent = input.value;
  }
}
for (const input of $$('[data-setting]')) {
  input.addEventListener('input', () => {
    state.settings[input.dataset.setting] = Number(input.value);
    input.nextElementSibling.textContent = input.value;
    state.unsaved = true;
    emit('settings');
  });
}
$('#resetSettings')?.addEventListener('click', () => {
  state.settings = { ...DEFAULT_SETTINGS };
  emit('settings');
});

function updateStale() {
  const stale = Boolean(state.model) &&
    ['blurSize', 'threshold', 'morphSize'].some(k => state.model.settings[k] !== state.settings[k]);
  for (const n of $$('[data-stale]')) n.hidden = !stale;
}

// ---- Step 8: build the dataset and train -------------------------------------

function averageSilhouette(items) {
  const out = makeGray(64, 64);
  if (!items.length) return out;
  const sum = new Float32Array(64 * 64);
  for (const item of items) item.normalized.data.forEach((v, i) => { sum[i] += v; });
  sum.forEach((v, i) => { out.data[i] = Math.round(v / items.length); });
  return out;
}

function statCard(value, label, tone = '') {
  return el('div', { class: 'stat ' + tone }, el('div', { class: 'stat-value', text: value }), el('div', { class: 'stat-label', text: label }));
}

function renderDataset(data) {
  const usable = label => data.kept.filter(k => k.photo.label === label).length;
  $('#trainStats').replaceChildren(
    statCard(String(usable('thumbs_up')), 'usable thumbs up photos'),
    statCard(String(usable('thumbs_down')), 'usable thumbs down photos'),
    statCard(String(data.rejected.length), 'rejected masks', data.rejected.length ? 'warn' : ''),
    statCard(`${data.features.length} × ${fmt(data.features[0]?.length ?? 1764)}`, 'feature array (photos × numbers)'),
  );
  const byId = new Map();
  for (const k of data.kept) byId.set(k.photo.id, { ...k, ok: true });
  for (const r of data.rejected) byId.set(r.photo.id, { ...r, ok: false });
  for (const [label, box] of [['thumbs_up', '#silUp'], ['thumbs_down', '#silDown']]) {
    $(box).replaceChildren(...photosOf('train', label).map(photo => {
      const item = byId.get(photo.id);
      if (!item) return null;
      return item.ok
        ? el('img', { class: 'sil', src: imageUrl(item.normalized), alt: 'silhouette' })
        : el('img', { class: 'sil rejected', src: photo.thumb, alt: 'rejected photo', title: item.reason });
    }));
  }
  paint($('#avgUp'), averageSilhouette(data.kept.filter(k => k.photo.label === 'thumbs_up')));
  paint($('#avgDown'), averageSilhouette(data.kept.filter(k => k.photo.label === 'thumbs_down')));
  $('#trainDataset').hidden = false;
  $('#trainResults').hidden = false;
}

function drawLossChart(history) {
  const svg = $('#lossChart');
  const W = 520, H = 220, L = 46, R = 14, T = 14, B = 38;
  const maxStep = Math.max(6, history.length - 1);
  const maxLoss = Math.max(0.7, ...history.map(h => h.loss));
  const x = s => L + (s / maxStep) * (W - L - R);
  const y = v => T + (1 - v / maxLoss) * (H - T - B);
  let html = '';
  for (let v = 0; v <= maxLoss + 1e-9; v += 0.1) {
    const yy = y(v).toFixed(1);
    html += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
    if (Math.round(v * 10) % 2 === 0) html += `<text class="tick" x="${L - 8}" y="${yy}" dy="4" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  for (let s = 0; s <= maxStep; s++) html += `<text class="tick" x="${x(s).toFixed(1)}" y="${H - B + 18}" text-anchor="middle">${s}</text>`;
  html += `<text class="axis" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">training step</text>`;
  html += `<text class="axis" transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">loss</text>`;
  if (history.length) {
    const points = history.map(h => `${x(h.step).toFixed(1)},${y(h.loss).toFixed(1)}`).join(' ');
    html += `<polyline class="loss" points="${points}"/>`;
    html += history.map(h => `<circle class="dot" cx="${x(h.step).toFixed(1)}" cy="${y(h.loss).toFixed(1)}" r="4"><title>Step ${h.step}: loss ${h.loss.toFixed(3)}</title></circle>`).join('');
  }
  svg.innerHTML = html;
}

function logLine(entry) {
  return el('li', {},
    el('b', { text: `Step ${entry.step}` }),
    entry.step === 0 ? ' (before learning)' : '',
    ` · loss ${entry.loss.toFixed(3)} · training accuracy ${pct(entry.accuracy)}`);
}

function renderLearned(m) {
  drawLossChart(m.history);
  $('#stepLog').replaceChildren(...m.history.map(logLine));
  const last = m.history[m.history.length - 1];
  const total = m.trainingCounts.thumbs_up + m.trainingCounts.thumbs_down;
  $('#trainAccuracy').textContent =
    `Training accuracy: ${pct(last.accuracy)} (${Math.round(last.accuracy * total)} / ${total}). ` +
    'This uses photos the model has already seen. Step 9 checks new ones.';
  const glyphs = weightGlyphs(m.coef, m.settings.featureSize);
  drawGlyphs($('#weightsUp'), glyphs.up, '#ffffff', 28);
  drawGlyphs($('#weightsDown'), glyphs.down, '#ffffff', 28);
  $('#trainLearned').hidden = false;
  $('#trainResults').hidden = false;
}

$('#trainButton').addEventListener('click', async () => {
  const button = $('#trainButton'), status = $('#trainStatus');
  const photos = photosOf('train');
  if (!photos.length) { status.textContent = 'Collect training photos in step 3 first.'; return; }
  button.disabled = true;
  try {
    const settings = { ...state.settings };
    const data = await buildDataset(photos, settings, async (done, total) => {
      status.textContent = `Processing photos: ${done} / ${total}`;
      await sleep(0);
    });
    $('#trainLearned').hidden = true;
    renderDataset(data);
    const usable = label => data.labels.filter(l => l === label).length;
    if (LABELS.some(label => usable(label) < 10)) {
      status.textContent = 'At least 10 usable training photos of each gesture are needed. Look at the rejected photos, adjust the mask settings (step 5) or collect more, then try again.';
      return;
    }
    const history = [];
    $('#stepLog').replaceChildren();
    $('#trainLearned').hidden = false;
    $('#trainAccuracy').textContent = '';
    for (const c of ['#weightsUp', '#weightsDown']) blank($(c));
    drawLossChart(history);
    const trained = await trainModel(data, settings, async entry => {
      history.push(entry);
      drawLossChart(history);
      $('#stepLog').append(logLine(entry));
      status.textContent = `Training, step ${entry.step}...`;
      await sleep(260);
    });
    state.model = trained;
    state.testReport = null;
    state.unsaved = true;
    renderLearned(trained);
    status.textContent = `Trained on ${trained.trainingCounts.thumbs_up} thumbs up and ${trained.trainingCounts.thumbs_down} thumbs down photos in ${trained.history.length - 1} steps. Next: step 9, test on new photos.`;
    $('#testResults').hidden = true;
    emit('model');
  } catch (error) {
    console.error(error);
    status.textContent = 'Something went wrong: ' + error.message;
  } finally {
    button.disabled = false;
  }
});

// ---- Step 9: evaluate on test photos -------------------------------------------

function resultCard(photo, normalized, lines, tone) {
  return el('figure', { class: 'card ' + (tone || '') },
    el('div', { class: 'card-images' },
      el('img', { src: photo.thumb, alt: 'test photo' }),
      normalized ? el('img', { class: 'px', src: imageUrl(normalized), alt: 'silhouette' }) : null),
    el('figcaption', {}, ...lines.flatMap((line, i) => (i ? [el('br'), line] : [line]))));
}

function renderTest(report) {
  const { correct, usable, total, matrix } = report;
  $('#testStats').replaceChildren(
    statCard(usable ? pct(correct / usable) : 'n/a', `classifier accuracy on usable masks (${correct} / ${usable})`),
    statCard(pct(correct / total), `complete pipeline (${correct} / ${total}); rejected masks count as wrong`),
    statCard(String(report.rejected.length), 'rejected masks', report.rejected.length ? 'warn' : ''),
  );
  const cell = (a, p) => el('td', { class: a === p ? 'diag' : '', text: String(matrix[a][p]) });
  $('#confusion').replaceChildren(
    el('tr', {}, el('th', {}), el('th', { text: 'predicted up' }), el('th', { text: 'predicted down' })),
    ...LABELS.map(a => el('tr', {}, el('th', { text: `actual ${SHORT[a]}` }), ...LABELS.map(p => cell(a, p)))),
  );
  const testPhotos = photosOf('test');
  $('#perLabel').replaceChildren(el('h3', { text: 'Per gesture' }), ...LABELS.map(label => {
    const raw = testPhotos.filter(p => p.label === label).length;
    const right = report.results.filter(r => r.actual === label && r.predicted === label).length;
    const rej = report.rejected.filter(r => r.photo.label === label).length;
    return el('p', {}, el('b', { text: NICE[label] }), `: ${right} / ${raw} correct overall · ${rej} mask${rej === 1 ? '' : 's'} rejected`);
  }));

  const wrong = report.results.filter(r => r.predicted !== r.actual);
  const shown = wrong.length ? wrong.slice(0, 12) : report.results.slice(0, 6);
  $('#mistakesTitle').textContent = wrong.length
    ? `Mistakes (${wrong.length})`
    : 'No classification mistakes in this test. Some examples:';
  $('#mistakes').replaceChildren(...shown.map(r => resultCard(r.photo, r.normalized, [
    `Actual: ${NICE[r.actual]}`,
    `Predicted: ${NICE[r.predicted]} (${Math.round(r.score * 100)}%)`,
  ], r.predicted === r.actual ? 'good' : 'bad')));
  $('#rejectedBlock').hidden = !report.rejected.length;
  $('#rejected').replaceChildren(...report.rejected.slice(0, 12).map(r =>
    resultCard(r.photo, r.normalized, [`Actual: ${NICE[r.photo.label]}`, `Rejected: ${r.reason}`], 'warn')));
  $('#testResults').hidden = false;
}

$('#testButton').addEventListener('click', async () => {
  const button = $('#testButton'), status = $('#testStatus');
  if (!state.model) { status.textContent = 'Train a model in step 8 first.'; return; }
  const photos = photosOf('test');
  if (LABELS.some(label => !photos.some(p => p.label === label))) {
    status.textContent = 'Collect test photos of both gestures first.';
    return;
  }
  button.disabled = true;
  try {
    const report = await evaluate(state.model, photos, async (done, total) => {
      status.textContent = `Processing test photos: ${done} / ${total}`;
      await sleep(0);
    });
    state.testReport = report;
    renderTest(report);
    status.textContent = report.usable
      ? `Tested on ${report.total} new photos.`
      : 'No test masks were usable. Check the test background and the mask settings.';
    emit('progress');
  } catch (error) {
    console.error(error);
    status.textContent = 'Something went wrong: ' + error.message;
  } finally {
    button.disabled = false;
  }
});

// ---- Step 10: live predictions ---------------------------------------------------

const live = { running: false, recent: [], frames: 0, startedAt: 0, timer: null };

function showPrediction(label, probabilityUp, message) {
  const labelNode = $('#predictionLabel');
  labelNode.className = 'prediction-label' + (label ? '' : ' muted');
  labelNode.textContent = label ? (label === 'thumbs_up' ? 'THUMBS UP' : 'THUMBS DOWN') : (message || 'no prediction yet');
  $('#predictionMeter').style.width = probabilityUp == null ? '50%' : `${(probabilityUp * 100).toFixed(1)}%`;
  $('#prediction').classList.toggle('idle', probabilityUp == null);
  $('#predictionScore').textContent = probabilityUp == null
    ? 'model score: n/a'
    : `model score: ${Math.round(Math.max(probabilityUp, 1 - probabilityUp) * 100)}% ${NICE[label]} (p(up) = ${probabilityUp.toFixed(2)})`;
}

function stopLive(message) {
  live.running = false;
  clearTimeout(live.timer);
  $('#liveButton').textContent = 'Start predictions';
  $('#liveButton').classList.remove('active');
  if (message) $('#liveStatus').textContent = message;
  emit('progress');
}

function liveStep() {
  if (!live.running) return;
  const started = performance.now();
  try {
    if (!cameraReady()) throw new Error('The camera stopped.');
    const m = state.model, bg = state.backgrounds.live;
    const shot = snapshot(m.camera.sideFraction, m.camera.imageSize);
    if (!sameSize(shot.frameSize, bg.frameSize)) throw new Error('The camera changed. Record the live background again.');
    const result = preprocess(shot.image, bg.image, m.settings);
    paint($('#liveMask'), result.mask);
    paint($('#liveInput'), result.normalized);
    if (result.valid) {
      live.recent.push(predict(m, hogFeatures(result.normalized)).probabilityUp);
      if (live.recent.length > 3) live.recent.shift();
      const average = live.recent.reduce((a, b) => a + b, 0) / live.recent.length;
      showPrediction(average > 0.5 ? 'thumbs_up' : 'thumbs_down', average);
    } else {
      live.recent = [];
      showPrediction(null, null, result.reason);
    }
    live.frames++;
    state.liveFrames++;
    const seconds = (performance.now() - live.startedAt) / 1000;
    $('#liveStatus').textContent = `Processed ${fmt(live.frames)} frames (about ${(live.frames / seconds).toFixed(0)} per second). Keep the camera and background still.`;
  } catch (error) {
    stopLive(error.message);
    return;
  }
  live.timer = setTimeout(liveStep, Math.max(0, 70 - (performance.now() - started)));
}

$('#liveButton').addEventListener('click', () => {
  const status = $('#liveStatus');
  if (live.running) { stopLive(`Stopped after ${fmt(live.frames)} frames.`); return; }
  if (!state.model) { status.textContent = 'Train a model in step 8 first.'; return; }
  if (!state.backgrounds.live) { status.textContent = 'Record the live background above first.'; return; }
  if (!cameraReady()) { status.textContent = 'Start the camera first.'; return; }
  Object.assign(live, { running: true, recent: [], frames: 0, startedAt: performance.now() });
  $('#liveButton').textContent = 'Stop predictions';
  $('#liveButton').classList.add('active');
  liveStep();
});
onCameraChange(c => { if (!c.stream && live.running) stopLive('The camera stopped.'); });

// ---- Step 11: save / load / start over ------------------------------------------

$('#saveButton').addEventListener('click', async () => {
  const button = $('#saveButton'), status = $('#saveStatus');
  if (!state.photos.length && !state.model) { status.textContent = 'Nothing to save yet.'; return; }
  button.disabled = true;
  try {
    status.textContent = 'Preparing the ZIP...';
    const blob = await saveProject(state, async (done, total) => {
      status.textContent = `Packing photos: ${done} / ${total}`;
      await sleep(0);
    });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:]/g, '').replace('T', '-');
    const link = el('a', { href: URL.createObjectURL(blob), download: `thumbs_project_${stamp}.zip` });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 20000);
    state.unsaved = false;
    state.saved = true;
    status.textContent = `Downloaded ${link.download} (${(blob.size / 1e6).toFixed(1)} MB).`;
    emit('progress');
  } catch (error) {
    console.error(error);
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

$('#loadInput').addEventListener('change', async event => {
  const input = event.target, status = $('#saveStatus');
  const file = input.files[0];
  input.value = '';
  if (!file) return;
  if ((state.photos.length || state.model) && !confirm('Loading replaces the photos and model in this tab. Continue?')) return;
  try {
    status.textContent = 'Loading...';
    const loaded = await loadProject(file);
    stopLive();
    for (const p of loaded.photos) p.thumb = imageUrl(p.image, 80);
    Object.assign(state, {
      settings: { ...DEFAULT_SETTINGS, ...loaded.settings },
      backgrounds: { train: null, test: null, live: null, ...loaded.backgrounds },
      photos: loaded.photos,
      model: loaded.model,
      testReport: null,
      exampleId: null,
      unsaved: false,
    });
    $('#trainResults').hidden = true;
    $('#trainDataset').hidden = true;
    $('#trainLearned').hidden = true;
    $('#testResults').hidden = true;
    if (state.model) {
      renderLearned(state.model);
      $('#trainStatus').textContent = 'Loaded a trained model. Click the button to retrain and see every processed photo.';
    }
    emit('settings', 'backgrounds', 'photos', 'model');
    status.textContent = `Loaded ${state.photos.length} photos${state.model ? ' and a trained model' : ''}.`;
  } catch (error) {
    console.error(error);
    status.textContent = 'Could not load that file: ' + error.message;
  }
});

$('#resetButton').addEventListener('click', () => {
  if (!confirm('Start over? This erases every photo, background and model in this tab.')) return;
  state.unsaved = false;
  location.reload();
});

window.addEventListener('beforeunload', event => {
  if (state.unsaved && state.photos.length) {
    event.preventDefault();
    event.returnValue = '';
  }
});

// ---- Progress in the step list ---------------------------------------------------

function updateProgress() {
  const hasExample = Boolean(state.exampleId);
  const trainCounts = LABELS.map(l => photosOf('train', l).length);
  const done = {
    camera: state.checked,
    background: Boolean(state.backgrounds.train),
    collect: trainCounts.every(n => n >= TARGET.train),
    grayscale: hasExample, mask: hasExample, silhouette: hasExample, features: hasExample,
    train: Boolean(state.model),
    test: Boolean(state.testReport),
    live: state.liveFrames > 0,
    save: state.saved,
  };
  const started = { collect: trainCounts.some(n => n > 0) };
  for (const link of $$('.toc a[data-step]')) {
    const step = link.dataset.step;
    link.classList.toggle('done', Boolean(done[step]));
    link.classList.toggle('started', !done[step] && Boolean(started[step]));
  }
}

const sectionLinks = new Map($$('.toc a[data-step]').map(a => [a.getAttribute('href').slice(1), a]));
const scrollSpy = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    for (const a of sectionLinks.values()) a.classList.remove('current');
    sectionLinks.get(entry.target.id)?.classList.add('current');
  }
}, { rootMargin: '-30% 0px -60% 0px' });
for (const id of sectionLinks.keys()) {
  const section = document.getElementById(id);
  if (section) scrollSpy.observe(section);
}

// ---- Wire the events together ---------------------------------------------------

on('photos', () => { renderPicker(); queueExplorer(); updateProgress(); });
on('settings', () => { syncSliders(); queueExplorer(); updateStale(); });
on('backgrounds', updateProgress);
on('model', () => { updateStale(); updateProgress(); });
on('progress', updateProgress);

syncSliders();
renderPicker();
renderExplorer();
showPrediction(null, null);
updateProgress();
