import 'package:flutter/services.dart';

/// 通过 MethodChannel 调用原生上色核心。
class ColorizeService {
  static const MethodChannel _channel =
      MethodChannel('com.kiastr.aicolorize/colorize');

  /// 对单张图片上色，返回上色后图片的本地路径。
  static Future<String> colorize({
    required String inputPath,
    required String modelPath,
    required String type,
    required bool useNnapi,
    String? outputPath,
  }) async {
    final String? out = await _channel.invokeMethod<String>('colorize', {
      'inputPath': inputPath,
      'modelPath': modelPath,
      'type': type,
      'useNnapi': useNnapi,
      if (outputPath != null) 'outputPath': outputPath,
    });
    if (out == null) throw Exception('上色失败：返回为空');
    return out;
  }
}
