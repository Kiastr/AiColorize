package com.kiastr.aicolorize

import android.content.Context
import android.os.Environment
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors

/**
 * MethodChannel 桥接：Dart 侧通过 'colorize' 方法调用原生计算核心。
 * 单次调用完成：输入图片 -> 预处理 -> ONNX 推理 -> 后处理 -> 输出图片路径。
 */
class ColorizePlugin(private val appContext: Context) : MethodChannel.MethodCallHandler {

    companion object {
        const val CHANNEL = "com.kiastr.aicolorize/colorize"

        fun registerWith(flutterEngine: FlutterEngine, appContext: Context) {
            val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            channel.setMethodCallHandler(ColorizePlugin(appContext))
        }
    }

    private val engine = ColorizeEngine()
    private val executor = Executors.newSingleThreadExecutor()
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "colorize") {
            result.notImplemented()
            return
        }
        val inputPath = call.argument<String>("inputPath")
        val modelPath = call.argument<String>("modelPath")
        val type = call.argument<String>("type") ?: "ddcolor"
        val useNnapi = call.argument<Boolean>("useNnapi") ?: true
        val outputPath = call.argument<String>("outputPath")

        if (inputPath == null || modelPath == null) {
            result.error("BAD_ARGS", "inputPath / modelPath 不能为空", null)
            return
        }

        // 推理在后台线程执行，避免阻塞 UI
        executor.execute {
            try {
                val inputFile = File(inputPath)
                val outputFile = if (outputPath != null) {
                    File(outputPath)
                } else {
                    // 默认保存到 app 外部 Pictures 目录（文件管理器可直接访问，无需 root）
                    // 路径: /storage/emulated/0/Android/data/com.kiastr.aicolorize/files/Pictures/AiColorize/
                    val outDir = File(
                        appContext.getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                        "AiColorize"
                    )
                    if (!outDir.exists()) outDir.mkdirs()
                    File(outDir, "${inputFile.nameWithoutExtension}_colorized.png")
                }
                val out = engine.colorize(inputFile, outputFile, modelPath, type, useNnapi)
                mainHandler.post { result.success(out.absolutePath) }
            } catch (e: Exception) {
                mainHandler.post {
                    result.error("COLORIZE_FAILED", e.message ?: "unknown error", null)
                }
            }
        }
    }
}
