"use client";
import { Jimp } from "jimp";
import { useState, useRef, useEffect } from "react";
import { Tensor, InferenceSession } from 'onnxruntime-web';
import Modal from './modal';
import nextConfig from '../next.config';
const BASE_PATH = nextConfig.basePath || "";

export default function Home() {
  const [imageSrc, setImageSrc] = useState<string>();
  const [downloadURL, setDownloadURL] = useState<string>("");
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onChangeImage = async (e: React.FormEvent<HTMLInputElement>) => {
    setDownloadURL("");
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const heic2any = require("heic2any");
    if (!e.currentTarget.files) return;
    const file = e.currentTarget.files[0];
    const reader = new FileReader();
    reader.onload = () => { setImageSrc(reader.result as string); }
    const fname = file.name.toLowerCase();
    if (fname.endsWith(".heic") || fname.endsWith(".heif")) {
      console.log("processing heic");
      const outputBlob = await heic2any({
        blob: file,
        toType: 'image/jpeg',
      });
      if (!Array.isArray(outputBlob)) {
        reader.readAsDataURL(outputBlob);
      }
    } else {
      reader.readAsDataURL(file);
    }
  };
  const processImage = async (path: string) => {
    const img = new window.Image();

    // draw image
    await new Promise<void>((resolve) => {
      img.onload = () => {
        const canvas = canvasRef.current!;
        const ctx = canvas.getContext("2d")!;
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        resolve();
      };
      img.src = path;
    });

    // set input tensor
    const {jimg, origWidth, origHeight} = await Jimp.read(path).then((imageBuffer) => {
      const origWidth = imageBuffer.width;
      const origHeight = imageBuffer.height;
      // return {jimg: imageBuffer.resize({
      //   w: origWidth - origWidth%32,
      //   h: origHeight - origHeight%32
      // }), origWidth, origHeight};
      return {jimg: imageBuffer.resize({h: 480, w: 960}), origWidth, origHeight};
    });

    // // debug
    // const palette = canvasRef.current!.getContext("2d")!.getImageData(0, 0, 960, 480);
    // palette.data.set(jimg.bitmap.data);
    // canvasRef.current!.getContext("2d")!.putImageData(palette, 0, 0);

    const imageBufferData = jimg.bitmap.data;
    const dims = [1, 3, jimg.height, jimg.width];
    const [redArray, greenArray, blueArray] = [new Array<number>(), new Array<number>(), new Array<number>()];
    for (let i = 0; i < imageBufferData.length; i += 4) {
      redArray.push(imageBufferData[i]);
      greenArray.push(imageBufferData[i + 1]);
      blueArray.push(imageBufferData[i + 2]);
      // skip data[i + 3] to filter out the alpha channel
    }
    // YOLOX requires BGR input image
    const transposedData = blueArray.concat(greenArray).concat(redArray);
    // float32 flattened [1, 3, H, W]
    const float32Data = Float32Array.from(transposedData);
    console.log(float32Data);
    const inputTensor = new Tensor("float32", float32Data, dims);
    // console.log('Input tensor created', dims, inputTensor.data.byteLength);

    // load session
    const session = await InferenceSession
                          .create(`${BASE_PATH}/yolox_t_body_head_hand_face_0299_0.4265_post_1x3x480x960.onnx`,
                          { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    console.log('Inference session created');

    // inference
    const feeds: Record<string, Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;
    const outputData = await session.run(feeds);
    const outputTensor = outputData[session.outputNames[0]];  // batchno_classid_score_x1y1x2y2: float32[boxes, 7], classid: [body, head, hand, face]
    const rawOutput = Array.prototype.slice.call(outputTensor.data);
    const faces = [];
    const THRESH = 0.35;
    for (let i = 0; i < rawOutput.length; i += 7) {
      const [_batchno, classid, score, x1, y1, x2, y2] = rawOutput.slice(i, i+7);
      if (score >= THRESH && classid == 1) {  // confident face
        faces.push([
          x1 * origWidth / jimg.width,
          y1 * origHeight / jimg.height,
          x2 * origWidth / jimg.width,
          y2 * origHeight / jimg.height,
        ]);
      }
    }
    console.log(faces);

    // // visualize
    // const ctx = canvasRef.current!.getContext("2d")!;
    // for (let face of faces) {
    //   ctx.strokeStyle = "red";
    //   ctx.lineWidth = 5;
    //   ctx.strokeRect(face[0], face[1], face[2]-face[0], face[3]-face[1]);
    // }

    // gaussian blur
    const kernel = [
      [1, 4, 6, 4, 1],
      [4, 16, 24, 16, 4],
      [6, 24, 36, 24, 6],
      [4, 16, 24, 16, 4],
      [1, 4, 6, 4, 1],
    ];
    const ctx = canvasRef.current!.getContext("2d")!;
    for (const face of faces) {
      const faceImg = await Jimp.fromBitmap(ctx.getImageData(face[0], face[1], face[2]-face[0], face[3]-face[1]));
      const faceW = faceImg.width;
      const faceH = faceImg.height;
      const faceImg2 = await faceImg.resize({w: 20, h: 20});
      const bitmap = faceImg2.bitmap.data;
      const blurred = Float32Array.from(bitmap);
      for (let x = 0; x < faceImg2.width; x++) {
        for (let y = 0; y < faceImg2.height; y++) {
          for (let off = 0; off < 3; off++) {
            let w = 0;
            let c = 0;
            for (let i = 0; i < 5; i++) {
              for (let j = 0; j < 5; j++) {
                if (x + i - 2 >= 0 && x + i - 2 < faceImg2.width && y + j - 2 >= 0 && y + j - 2 < faceImg2.height) {
                  w += bitmap[(y + j - 2) * faceImg2.width * 4 + (x + i - 2) * 4 + off] * kernel[i][j];
                  c += kernel[i][j];
                }
              }
            }
            blurred[y * faceImg2.width * 4 + x * 4 + off] = w / c;
          }
        }
      }
      faceImg2.bitmap.data = blurred;
      const faceImg3 = await faceImg2.resize({w: faceW, h: faceH});
      const imageData = new ImageData(
        new Uint8ClampedArray(faceImg3.bitmap.data),
        faceImg3.bitmap.width,
        faceImg3.bitmap.height
      );
      ctx.putImageData(imageData, face[0], face[1]);
    }

    // download image
    setDownloadURL(canvasRef.current!.toDataURL("image/jpeg"));
  };
  useEffect(() => {
    if (!imageSrc) return;
    processImage(imageSrc);
  }, [imageSrc]);

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
        <h1 className="text-4xl font-extrabold">FaceBlur</h1>
        Hide their faces.

        <div className="flex flex-col items-center justify-center">
          <label className="w-64 flex flex-col items-center px-4 py-6 bg-white rounded-lg shadow-lg tracking-wide hover:bg-blue-700 text-black hover:text-white">
            <span className="">Select an image</span>
            <input type="file" className="hidden" onChange={onChangeImage} />
          </label>
          {!downloadURL && imageSrc && (
            <div className="relative mt-4">
              <canvas ref={canvasRef} className="max-w-full h-auto" />
              {!downloadURL && (
                <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-75">
                  <div className="w-16 h-16 border-4 border-blue-500 rounded-full spinner" />
                </div>
              )}
            </div>
          )}
          {downloadURL && (
            <div className="relative mt-4">
              <img
                className="max-w-full h-auto"
                src={downloadURL}
                alt="generated Image"
              />
            </div>
          )}
        </div>
      </main>
      <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
        <button onClick={() => setIsModalOpen(true)} className="text-blue-500 underline">
          Credits
        </button>
        <Modal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
      </footer>
    </div>
  );
}
