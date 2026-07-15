/// 模型配置
///  - 三个引擎选项：DeOldify fp32（高质量）/ DeOldify int8（省流量）/ DDColor int8（漫画风）
///  - 主源统一用本仓库专用 models release（Kiastr/AiColorize），第三方源作备用
///  - 所有模型 I/O spec 已验证与 pipeline 一致
class ModelSource {
  final String name;
  final String url;
  const ModelSource(this.name, this.url);
}

class ModelConfig {
  final String key; // 'deoldify_fp32' | 'deoldify_int8' | 'ddcolor'
  final String filename; // 下载后保存的文件名
  final String label;
  final List<ModelSource> sources;
  const ModelConfig({
    required this.key,
    required this.filename,
    required this.label,
    required this.sources,
  });
}

const String _repo =
    'https://github.com/Kiastr/AiColorize/releases/download/models';

const Map<String, ModelConfig> modelConfigs = {
  'deoldify_fp32': ModelConfig(
    key: 'deoldify_fp32',
    filename: 'deoldify_fp32.onnx',
    label: 'DeOldify (fp32, 243MB, 高质量)',
    sources: [
      ModelSource('AiColorize 仓库', '$_repo/deoldify_fp32.onnx'),
      ModelSource(
        'GitHub instant-high (备用)',
        'https://github.com/instant-high/deoldify-onnx/releases/download/deoldify-onnx/deoldify.onnx',
      ),
    ],
  ),
  'deoldify_int8': ModelConfig(
    key: 'deoldify_int8',
    filename: 'deoldify_int8.onnx',
    label: 'DeOldify (int8, 61MB, 省流量)',
    sources: [
      ModelSource('AiColorize 仓库', '$_repo/deoldify_int8.onnx'),
      ModelSource(
        'GitHub MartinDelophy (备用)',
        'https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/deoldify.quant.onnx',
      ),
    ],
  ),
  'ddcolor': ModelConfig(
    key: 'ddcolor',
    filename: 'ddcolor-int8.onnx',
    label: 'DDColor (int8, 59MB, 漫画风)',
    sources: [
      ModelSource('AiColorize 仓库', '$_repo/ddcolor-int8.onnx'),
      ModelSource(
        'HuggingFace Faridzar (备用)',
        'https://huggingface.co/Faridzar/ddcolor-mirror/resolve/main/ddcolor-int8.onnx',
      ),
    ],
  ),
};
