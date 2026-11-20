# thumbs-up-down-cv-lab

A browser version of the "teach your webcam to recognize thumbs up and thumbs down" computer vision lab.

**Open the lab:** https://pranavakcode.github.io/thumbs-up-down-cv-lab/

The original Colab notebook needed a Python runtime and Colab's camera bridge, which often hit camera permission errors inside notebook outputs. This version does the same steps in plain JavaScript, in a normal web page, so the camera prompt is the browser's own and nothing has to be installed or uploaded.

## What students do

1. Open the webcam (or the practice camera if theirs is blocked)
2. Record an empty background (median of 7 frames)
3. Collect 60 labeled photos of each gesture
4. Grayscale
5. Blur, background difference, threshold, opening/closing (with sliders and a histogram)
6. Largest connected region, crop, normalize to 64 × 64
7. HOG features (1,764 numbers)
8. Standardize + logistic regression, with the loss shown step by step and the learned weights drawn
9. Fresh test recording, accuracy, confusion matrix, mistakes
10. Live predictions with mask and classifier input shown beside the camera
11. Download / load the project as a ZIP

Every step has a "Show the code" toggle that prints the exact function the page is running.

## Practice camera

If a school device blocks the webcam, "Use practice camera" draws a moving hand on a canvas and feeds it to the page as a video stream. The top bar switches it between empty, thumbs up and thumbs down, so every step still works.

## Files

| file | what it does |
| --- | --- |
| `index.html` | the lab page and its text |
| `js/imgproc.js` | grayscale, blur, difference, threshold, morphology, connected components, normalization |
| `js/hog.js` | gradients and HOG features (plus drawing helpers) |
| `js/model.js` | standardization and logistic regression (Newton's method, balanced classes, C = 1) |
| `js/pipeline.js` | dataset building, training, prediction, evaluation |
| `js/camera.js`, `js/practice.js` | webcam access and the practice camera |
| `js/storage.js` | ZIP save/load (uses JSZip from cdnjs) |

## Run locally

Any static file server works. Camera access needs `localhost` or `https`.

```
python -m http.server 8000
```

Then open http://localhost:8000.

## Privacy

Photos stay in the browser tab's memory. Nothing is sent to a server. Refreshing the page clears them unless the project was downloaded.
