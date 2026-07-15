/// 模型配置：关键修正
///  - 桌面版 README 的 DeOldify 链接已失效；安卓实测可用的是 int8 量化版
///    deoldify.quant.onnx（体积更小、移动端更友好，已用 Python 跑通验证）。
///  - DDColor 原 piddnad github release (ddcolor_tiny.onnx) 已 404 失效；
///    改用 HuggingFace Faridzar 镜像的 ddcolor-int8.onnx（59MB，int8 量化，
///    I/O spec [1,3,256,256]->[1,2,256,256] 已验证与 pipeline 一致，
///    全 pipeline 实测 colorfulness≈60，与完整版近乎一致）。
class ModelSource {
  final String name;
  final String url;
  const ModelSource(this.name, this.url);
}

class ModelConfig {
  final String key; // 'deoldify' | 'ddcolor'
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

const Map<String, ModelConfig> modelConfigs = {
  'deoldify': ModelConfig(
    key: 'deoldify',
    filename: 'deoldify.quant.onnx',
    label: 'DeOldify (int8 量化, 推荐)',
    sources: [
      ModelSource(
        'GitHub MartinDelophy',
        'https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/deoldify.quant.onnx',
      ),
    ],
  ),
  'ddcolor': ModelConfig(
    key: 'ddcolor',
    filename: 'ddcolor.int8.onnx',
    label: 'DDColor (int8 量化, 推荐)',
    sources: [
      ModelSource(
        'HuggingFace Faridzar (int8, 59MB)',
        'https://huggingface.co/Faridzar/ddcolor-mirror/resolve/main/ddcolor-int8.onnx',
      ),
      ModelSource(
        'HuggingFace facefusion (完整版, 934MB)',
        'https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx',
      ),
    ],
  ),
};
