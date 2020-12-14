import { Router } from 'express';
import '@tensorflow/tfjs-node';
import * as faceapi from 'face-api.js';
import * as canvas from 'canvas';
import { faceDetectionNet, faceDetectionOptions} from './faceDetection';
import fs from 'fs';
import path from 'path';

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });


const router = Router();


router.get('/face', async (req, res) => {
  try {
    // res.sendFile(path.resolve(__dirname, './index.html'));
  await faceDetectionNet.loadFromDisk(path.resolve(__dirname, '../weights'));
  const img = await canvas.loadImage('./models/face1.jpg');
  const detections = await faceapi.detectSingleFace(img, faceDetectionOptions);
  console.log(typeof detections);
  const value = JSON.parse(JSON.stringify(detections));
  console.log('score: ', value._score);
  
  const out = await faceapi.createCanvasFromMedia(img);
  await faceapi.draw.drawDetections(out, detections);

  fs.writeFileSync('faceDetection.jpg', out.toBuffer('image/jpeg'));
  console.log('done, saved results to out/faceDetection.jpg');
  res.send(`image score: ${value._score}`);
  } catch (error) {
    console.log(error);
    console.log('problem with evaluating image, score too low to compute');
  }
});

export default router;