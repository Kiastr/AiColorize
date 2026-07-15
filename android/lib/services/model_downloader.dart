import 'package:dio/dio.dart';
import '../models/model_config.dart';

/// 模型下载器：多镜像源 fallback + 进度回调 + 取消。
class ModelDownloader {
  final Dio _dio = Dio();
  final CancelToken _cancelToken = CancelToken();

  Future<void> download({
    required ModelConfig config,
    required String savePath,
    required void Function(int received, int total) onProgress,
  }) async {
    for (final source in config.sources) {
      try {
        await _dio.download(
          source.url,
          savePath,
          cancelToken: _cancelToken,
          onReceiveProgress: (count, total) {
            onProgress(count, total);
          },
        );
        return;
      } catch (e) {
        // 当前源失败，尝试下一个镜像源
        continue;
      }
    }
    throw Exception('所有镜像源均下载失败，请检查网络或手动下载模型');
  }

  void cancel() {
    if (!_cancelToken.isCancelled) {
      _cancelToken.cancel('用户取消');
    }
  }
}
