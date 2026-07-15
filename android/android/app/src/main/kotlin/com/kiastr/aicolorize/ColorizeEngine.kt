package com.kiastr.aicolorize

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Collections

/**
 * 上色计算核心。两个函数严格 1:1 对齐桌面版 colorize.py（已由 Python 参考实现验证）：
 *  - DeOldify 输入值域 0–255（不归一化），输出完整 BGR
 *  - DDColor  输入值域 0–1（/255），输出仅 ab 两通道，需与原图 L 拼接
 *  - OpenCV float-LAB 真实范围 L∈[0,100]、a,b∈[−128,127]，必须用 OpenCV 转换
 */
class ColorizeEngine {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val modelManager = ModelManager(env)

    fun colorize(
        inputFile: File,
        outputFile: File,
        modelPath: String,
        type: String,
        useNnapi: Boolean
    ): File {
        val bitmap = BitmapFactory.decodeFile(inputFile.absolutePath)
            ?: throw IllegalArgumentException("无法解码图片: ${inputFile.absolutePath}")
        val session = modelManager.getSession(modelPath, useNnapi)
        val outBitmap = if (type == "ddcolor") {
            colorizeDdcolor(bitmap, session)
        } else {
            colorizeDeoldify(bitmap, session)
        }
        outBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputFile.outputStream())
        bitmap.recycle()
        outBitmap.recycle()
        return outputFile
    }

    // ----------------------------------------------------------------
    // DeOldify：输入 0–255，输出完整 BGR，再与原图 L 通道在 LAB 合并
    // 对应 colorize_deoldify（py_ref_impl.colorize_deoldify）
    // ----------------------------------------------------------------
    private fun colorizeDeoldify(inputBitmap: Bitmap, session: OrtSession): Bitmap {
        val originalBgr = ImageUtils.bitmapToBgrMat(inputBitmap) // (H,W,3) BGR uint8
        val h = originalBgr.height()
        val w = originalBgr.width()

        // target_l = 原图 BGR 的 B 通道（cv2.split 第一个）
        val targetL = Mat()
        Core.extractChannel(originalBgr, targetL, 0)

        // gray = BGR2GRAY
        val gray = Mat()
        Imgproc.cvtColor(originalBgr, gray, Imgproc.COLOR_BGR2GRAY)
        // gray_rgb = GRAY2RGB
        val grayRgb = Mat()
        Imgproc.cvtColor(gray, grayRgb, Imgproc.COLOR_GRAY2RGB)
        // resize(256,256)
        val input256 = Mat()
        Imgproc.resize(grayRgb, input256, Size(256.0, 256.0))
        // -> float32 0–255（不 /255！）
        val inputF = Mat()
        input256.convertTo(inputF, CvType.CV_32F)

        // NCHW [1,3,256,256] 0–255
        val buf = ImageUtils.hwcToNchwFloatBuffer(inputF)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, 256, 256))
        val results = session.run(Collections.singletonMap(inputName, tensor))
        val outputName = session.outputNames.iterator().next()
        val outValue = results[outputName]!!
        val outBuf = (outValue as OnnxTensor).floatBuffer // (1,3,256,256) NCHW BGR 0–255
        val colorized256 = ImageUtils.nchwToHwcMat(outBuf, 3, 256, 256) // (256,256,3) BGR
        tensor.close()
        results.close()

        // BGR2RGB -> uint8
        val colorizedRgb = Mat()
        Imgproc.cvtColor(colorized256, colorizedRgb, Imgproc.COLOR_BGR2RGB)
        colorized256.release()
        val colorizedUint8 = Mat()
        colorizedRgb.convertTo(colorizedUint8, CvType.CV_8U) // 饱和截断
        colorizedRgb.release()

        // resize 回原尺寸
        val colorizedFull = Mat()
        Imgproc.resize(colorizedUint8, colorizedFull, Size(w.toDouble(), h.toDouble()))
        colorizedUint8.release()

        // GaussianBlur(13,13)
        val blurred = Mat()
        Imgproc.GaussianBlur(colorizedFull, blurred, Size(13.0, 13.0), 0.0)
        colorizedFull.release()

        // ★ 刻意对齐原版：colorized 此时是 RGB，但原代码用 COLOR_BGR2LAB 处理（即把 RGB 当 BGR）
        val lab = Mat()
        Imgproc.cvtColor(blurred, lab, Imgproc.COLOR_BGR2Lab)
        blurred.release()

        val channels = ArrayList<Mat>()
        Core.split(lab, channels)
        val a = channels[1]
        val b = channels[2]
        val merged = ArrayList<Mat>()
        merged.add(targetL)
        merged.add(a)
        merged.add(b)
        val resultLab = Mat()
        Core.merge(merged, resultLab)
        lab.release()
        channels[0].release()
        a.release()
        b.release()

        // LAB2BGR
        val resultBgr = Mat()
        Imgproc.cvtColor(resultLab, resultBgr, Imgproc.COLOR_Lab2BGR)
        resultLab.release()
        targetL.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(resultBgr) // 内部 BGR2RGBA
        resultBgr.release()
        originalBgr.release()
        gray.release()
        grayRgb.release()
        input256.release()
        inputF.release()
        return outBitmap
    }

    // ----------------------------------------------------------------
    // DDColor：输入 0–1，输出仅 ab 两通道，需与原图 L 拼接
    // 对应 colorize_ddcolor_tiny（py_ref_impl.colorize_ddcolor）
    // ----------------------------------------------------------------
    private fun colorizeDdcolor(inputBitmap: Bitmap, session: OrtSession): Bitmap {
        val bgr = ImageUtils.bitmapToBgrMat(inputBitmap) // BGR uint8
        val h = bgr.height()
        val w = bgr.width()

        // img_norm = bgr / 255
        val imgNorm = Mat()
        bgr.convertTo(imgNorm, CvType.CV_32F, 1.0 / 255.0)

        // orig_l = BGR2Lab[:,:,:1]（float LAB, L∈[0,100]）
        val labFull = Mat()
        Imgproc.cvtColor(imgNorm, labFull, Imgproc.COLOR_BGR2Lab)
        imgNorm.release()
        val origL = Mat()
        Core.extractChannel(labFull, origL, 0)
        labFull.release()

        // img_resized 256
        val imgResized = Mat()
        Imgproc.resize(imgNorm, imgResized, Size(256.0, 256.0))
        // img_l from resized
        val labResized = Mat()
        Imgproc.cvtColor(imgResized, labResized, Imgproc.COLOR_BGR2Lab)
        val imgL = Mat()
        Core.extractChannel(labResized, imgL, 0)
        labResized.release()
        imgResized.release()

        // gray_lab = concat(img_l, 0, 0)
        val zeros = Mat(imgL.size(), imgL.type(), Scalar(0.0))
        val grayLabList = ArrayList<Mat>()
        grayLabList.add(imgL)
        grayLabList.add(zeros)
        grayLabList.add(zeros)
        val grayLab = Mat()
        Core.merge(grayLabList, grayLab)
        // gray_rgb = LAB2RGB（0–1）
        val grayRgb = Mat()
        Imgproc.cvtColor(grayLab, grayRgb, Imgproc.COLOR_Lab2RGB)
        zeros.release()

        // NCHW [1,3,256,256] 0–1
        val buf = ImageUtils.hwcToNchwFloatBuffer(grayRgb)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, 256, 256))
        val results = session.run(Collections.singletonMap(inputName, tensor))
        val outputName = session.outputNames.iterator().next()
        val outValue = results[outputName]!!
        val outBuf = (outValue as OnnxTensor).floatBuffer // (1,2,256,256) NCHW ab
        val ab256 = ImageUtils.nchwToHwcMat(outBuf, 2, 256, 256) // (256,256,2) ab
        tensor.close()
        results.close()

        // resize ab 回原尺寸
        val abFull = Mat()
        Imgproc.resize(ab256, abFull, Size(w.toDouble(), h.toDouble()))
        ab256.release()

        // output_lab = concat(orig_l, abFull)
        val outLabList = ArrayList<Mat>()
        outLabList.add(origL)
        outLabList.add(abFull)
        val outLab = Mat()
        Core.merge(outLabList, outLab)
        origL.release()
        abFull.release()

        // output_bgr = LAB2BGR（float 0–1）
        val outBgr = Mat()
        Imgproc.cvtColor(outLab, outBgr, Imgproc.COLOR_Lab2BGR)
        outLab.release()

        // (outBgr * 255).round().clip(0,255).astype(uint8)
        // OpenCV convertTo(CV_8U) 内部用 cvRound 四舍五入 + 饱和截断（已实证），
        // 因此只需 *255 后直接 convertTo，切勿再 +0.5（会变成双重舍入、整体偏亮 ~1）
        val scaled = Mat()
        Core.multiply(outBgr, Scalar(255.0), scaled)
        val outUint8 = Mat()
        scaled.convertTo(outUint8, CvType.CV_8U) // cvRound + 饱和截断 = clip[0,255]
        scaled.release()
        outBgr.release()

        val outBitmap = ImageUtils.bgrMatToBitmap(outUint8) // 内部 BGR2RGBA
        outUint8.release()
        bgr.release()
        grayLab.release()
        grayRgb.release()
        imgL.release()
        return outBitmap
    }

    fun close() {
        modelManager.close()
    }
}
