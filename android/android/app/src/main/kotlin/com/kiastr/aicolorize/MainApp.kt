package com.kiastr.aicolorize

import android.app.Application
import android.util.Log
import org.opencv.android.OpenCVLoader

/**
 * 应用入口：初始化 OpenCV 原生库。
 * OpenCV 4.x Maven AAR 通过 OpenCVLoader.initDebug() 加载内置 .so。
 */
class MainApp : Application() {
    override fun onCreate() {
        super.onCreate()
        if (!OpenCVLoader.initDebug()) {
            Log.w("AiColorize", "OpenCV initDebug() 失败，OpenCV 相关功能将不可用")
        }
    }
}
