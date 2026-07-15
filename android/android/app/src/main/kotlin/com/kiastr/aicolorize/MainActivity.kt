package com.kiastr.aicolorize

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        // 注册原生上色插件（MethodChannel）
        ColorizePlugin.registerWith(flutterEngine, applicationContext)
    }
}
