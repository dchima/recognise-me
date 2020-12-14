import { Router } from 'express';
import '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import * as canvas from 'canvas';
import * as faceapi from 'face-api.js';
import { faceDetectionNet, faceDetectionOptions} from './faceDetection';

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });


const router = Router();


router.get('/face', async (req, res) => {
  try {
    // res.sendFile(path.resolve(__dirname, './index.html'));
  await faceDetectionNet.loadFromDisk(path.resolve(__dirname, '../weights'));
  const img = await canvas.loadImage('./models/dog.jpg');
  const detections = await faceapi.detectSingleFace(img, faceDetectionOptions);
  console.log(typeof detections);
  const value = JSON.parse(JSON.stringify(detections));
  console.log('score: ', value._score);
  
  const out = await faceapi.createCanvasFromMedia(img);
  await faceapi.draw.drawDetections(out, detections);
  // if (detections[0])

  fs.writeFileSync('faceDetection.jpg', out.toBuffer('image/jpeg'));
  console.log('done, saved results to out/faceDetection.jpg');
  } catch (error) {
    console.log(error);
    console.log('problem with evaluating image, score too low to compute');
  }
});

export default router;