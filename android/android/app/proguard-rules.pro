# ONNX Runtime：保留全部类、字段、方法（含 JNI 桥接与枚举，NNAPI/CPU provider 用到反射）
-keep class ai.onnxruntime.** { *; }
-keepclassmembers enum ai.onnxruntime.** { *; }

# OpenCV：保留全部类与方法（含 native 方法签名、OpenCVLoader.initDebug）
-keep class org.opencv.** { *; }

# 本项目原生桥接代码（MethodChannel / ColorizeEngine / ModelManager 等）
-keep class com.kiastr.aicolorize.** { *; }

# 保留所有 native 方法签名（JNI 注册依赖方法名不被混淆）
-keepclasseswithmembernames class * {
    native <methods>;
}

# 抑制可选/传递依赖的缺失告警，避免 R8 触发 warning-as-error 中断构建
-dontwarn org.opencv.**
-dontwarn ai.onnxruntime.**
-dontwarn org.tensorflow.**
-dontwarn com.microsoft.**
