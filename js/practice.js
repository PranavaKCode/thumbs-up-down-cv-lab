// A pretend webcam: a drawn hand in front of a wall, for students whose
// camera is missing or blocked. It produces a real video stream, so every
// later step treats it exactly like a webcam.

const W = 640, H = 480;
const SKIN = '#e6b89c', CREASE = 'rgba(160, 96, 72, 0.45)', NAIL = '#f3d4c2';

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function makeBackdrop() {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const b = c.getContext('2d');
  const wall = b.createLinearGradient(0, 0, W, H);
  wall.addColorStop(0, '#62738c');
  wall.addColorStop(1, '#4b5b72');
  b.fillStyle = wall;
  b.fillRect(0, 0, W, H);
  const glow = b.createRadialGradient(W * 0.3, H * 0.15, 10, W * 0.3, H * 0.15, 420);
  glow.addColorStop(0, 'rgba(255,255,255,0.16)');
  glow.addColorStop(1, 'rgba(255,255,255,0)');
  b.fillStyle = glow;
  b.fillRect(0, 0, W, H);
  // Things around the edges, outside the green square.
  b.fillStyle = '#394456'; b.fillRect(0, 0, 64, H);                    // door frame
  b.fillStyle = '#8a6d4d'; b.fillRect(500, 70, 140, 12);               // shelf
  b.fillStyle = '#c9b48a'; b.fillRect(522, 26, 26, 44);                // books
  b.fillStyle = '#6c8f6a'; b.fillRect(552, 36, 22, 34);
  b.fillStyle = '#a45d5d'; b.fillRect(578, 30, 18, 40);
  return c;
}

function makeNoise() {
  // Real cameras flicker a little; this adds a faint, changing grain.
  const c = document.createElement('canvas');
  c.width = W / 4; c.height = H / 4;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(c.width, c.height);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() < 0.5 ? 0 : 255;
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
    img.data[i + 3] = Math.random() * 14;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

export function createPracticeScene() {
  const canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const backdrop = makeBackdrop();
  const noise = Array.from({ length: 4 }, makeNoise);
  const start = performance.now();
  let gesture = 'none', tick = 0;

  function drawHand(t) {
    const cx = W / 2 + 40 * Math.sin(t * 0.55) + 10 * Math.sin(t * 1.7);
    const cy = H / 2 - 6 + 26 * Math.sin(t * 0.8 + 1);
    const scale = 1 + 0.13 * Math.sin(t * 0.45 + 2);
    const tilt = 0.2 * Math.sin(t * 0.65);

    // Forearm, coming in from the bottom left.
    ctx.strokeStyle = SKIN;
    ctx.lineCap = 'round';
    ctx.lineWidth = 74 * scale;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy + 24);
    ctx.lineTo(cx - 160, H + 140);
    ctx.stroke();

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(tilt + (gesture === 'thumbs_down' ? Math.PI : 0));
    ctx.scale(scale, scale);
    ctx.fillStyle = SKIN;
    roundRect(ctx, -56, -44, 112, 96, 30);                 // fist
    ctx.fill();
    roundRect(ctx, -58, -120, 36, 96, 18);                 // thumb
    ctx.fill();
    ctx.fillStyle = NAIL;
    roundRect(ctx, -52, -115, 24, 22, 10);
    ctx.fill();
    ctx.strokeStyle = CREASE;                              // curled fingers
    ctx.lineWidth = 3;
    for (const y of [-18, 6, 30]) {
      ctx.beginPath();
      ctx.moveTo(-10, y);
      ctx.quadraticCurveTo(20, y + 5, 50, y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function frame() {
    const t = (performance.now() - start) / 1000;
    ctx.drawImage(backdrop, 0, 0);
    if (gesture !== 'none') drawHand(t);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(noise[tick++ % noise.length], 0, 0, W, H);
    ctx.imageSmoothingEnabled = true;
  }

  frame();
  const timer = setInterval(frame, 33);
  const stream = canvas.captureStream(30);
  return {
    stream,
    setGesture(next) { gesture = next; },
    stop() { clearInterval(timer); stream.getTracks().forEach(track => track.stop()); },
  };
}
