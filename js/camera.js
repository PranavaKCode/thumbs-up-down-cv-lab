// One shared camera for the whole page: a real webcam or the practice camera.
import { createPracticeScene } from './practice.js';

export const ROI_FRACTION = 0.65;   // the green square is 65% of the shorter side
export const IMAGE_SIZE = 224;      // saved photos are 224 x 224 pixels

export const camera = { stream: null, kind: 'off', gesture: 'none', devices: [], deviceId: null, error: null };

const listeners = new Set();
let practice = null;

// A hidden <video> that the snapshots are taken from.
const source = document.createElement('video');
source.muted = true;
source.playsInline = true;
source.setAttribute('playsinline', '');
Object.assign(source.style, { position: 'fixed', left: '-10000px', top: '0', width: '320px', pointerEvents: 'none' });
document.body.append(source);

const grabCanvas = document.createElement('canvas');
grabCanvas.width = grabCanvas.height = IMAGE_SIZE;
const grab = grabCanvas.getContext('2d', { willReadFrequently: true });

export function onCameraChange(fn) {
  listeners.add(fn);
  fn(camera);
  return () => listeners.delete(fn);
}

function emit() {
  for (const fn of listeners) fn(camera);
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

const PROBLEMS = {
  NotAllowedError: 'Camera permission was denied. Click the camera icon in the address bar (or your browser\'s site settings), allow the camera, and try again. If your school blocks the camera, use the practice camera instead.',
  PermissionDeniedError: 'Camera permission was denied. Allow the camera in your browser\'s site settings and try again.',
  NotFoundError: 'No webcam was found. Connect or enable a camera, or use the practice camera.',
  DevicesNotFoundError: 'No webcam was found. Connect or enable a camera, or use the practice camera.',
  NotReadableError: 'The webcam could not be read. Close other apps or tabs that are using it (Zoom, Meet, Photo Booth...) and try again.',
  TrackStartError: 'The webcam may be busy in another app. Close that app and try again.',
  OverconstrainedError: 'This camera can\'t provide the requested settings. Try another camera.',
  AbortError: 'Camera startup was interrupted. Try again.',
  SecurityError: 'The browser\'s security settings block the camera on this page. Use the practice camera, or ask your teacher.',
  InsecureContext: 'Camera access needs a secure (https://) page. Open the lab from its GitHub Pages address.',
  UnsupportedBrowser: 'This browser does not offer webcam access. Try a current Chrome, Edge, Firefox or Safari.',
  CameraDisconnected: 'The camera disconnected. Reconnect it and click Start camera again.',
  NoFrames: 'The webcam opened but sent no picture. Close other camera apps and try again.',
};

export function describeCameraError(error) {
  return PROBLEMS[error?.name] || 'Camera error: ' + (error?.message || String(error));
}

async function attach(stream, kind) {
  camera.stream = stream;
  camera.kind = kind;
  source.srcObject = stream;
  await source.play().catch(() => {});
  const deadline = Date.now() + 10000;
  while (!source.videoWidth || source.readyState < 2) {
    if (camera.stream !== stream) return;
    if (Date.now() > deadline) throw { name: 'NoFrames' };
    await sleep(50);
  }
}

function release() {
  if (camera.stream) camera.stream.getTracks().forEach(track => track.stop());
  if (practice) practice.stop();
  practice = null;
  source.srcObject = null;
  camera.stream = null;
  camera.kind = 'off';
}

export async function startWebcam(deviceId = null) {
  release();
  camera.error = null;
  camera.kind = 'starting';
  emit();
  try {
    if (!window.isSecureContext) throw { name: 'InsecureContext' };
    if (!navigator.mediaDevices?.getUserMedia) throw { name: 'UnsupportedBrowser' };
    const size = { width: { ideal: 640 }, height: { ideal: 480 } };
    const stream = await navigator.mediaDevices.getUserMedia({
      video: deviceId ? { deviceId: { exact: deviceId }, ...size } : { facingMode: 'user', ...size },
      audio: false,
    });
    await attach(stream, 'webcam');
    camera.deviceId = stream.getVideoTracks()[0]?.getSettings().deviceId ?? null;
    camera.devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => {
      if (camera.stream !== stream) return;
      release();
      camera.error = describeCameraError({ name: 'CameraDisconnected' });
      emit();
    }));
  } catch (error) {
    release();
    camera.error = describeCameraError(error);
  }
  emit();
}

export async function startPracticeCamera() {
  release();
  camera.error = null;
  practice = createPracticeScene();
  practice.setGesture(camera.gesture);
  try {
    await attach(practice.stream, 'practice');
  } catch (error) {
    release();
    camera.error = describeCameraError(error);
  }
  emit();
}

export function stopCamera() {
  release();
  emit();
}

export function setPracticeGesture(gesture) {
  camera.gesture = gesture;
  if (practice) practice.setGesture(gesture);
  emit();
}

export function cameraReady() {
  return Boolean(camera.stream) && source.readyState >= 2 && source.videoWidth > 0;
}

export function snapshot(sideFraction = ROI_FRACTION, size = IMAGE_SIZE) {
  // Crop the green square from the middle of the frame, mirror it like the
  // preview, and shrink it to size x size pixels.
  if (!cameraReady()) throw new Error('The camera is not ready. Click Start camera first.');
  const width = source.videoWidth, height = source.videoHeight;
  const side = Math.floor(Math.min(width, height) * sideFraction);
  const x = Math.floor((width - side) / 2), y = Math.floor((height - side) / 2);
  grab.save();
  grab.setTransform(-1, 0, 0, 1, size, 0);           // mirror left <-> right
  grab.drawImage(source, x, y, side, side, 0, 0, size, size);
  grab.restore();
  return { image: grab.getImageData(0, 0, size, size), frameSize: [height, width] };
}
