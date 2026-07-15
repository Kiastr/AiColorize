/// 模型配置
///  - 三个引擎选项：DeOldify fp32（高质量）/ DeOldify int8（省流量）/ DDColor int8（漫画风）
///  - 下载源：GitHub 直连 + 多个国内镜像（ghproxy / gh-proxy / cors.isteed）+ 第三方备用
///    任一源失败自动切换下一个，方便国内用户
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

/// GitHub 直连 + 国内镜像前缀（实测可用）
const List<String> _mirrorPrefixes = [
  '', // GitHub 直连
  'https://ghproxy.net/',
  'https://gh-proxy.com/',
  'https://cors.isteed.cc/',
];

/// 为指定文件生成 [直连 + 镜像] 源列表
List<ModelSource> _githubSources(String filename) {
  return _mirrorPrefixes.map((p) {
    final name = p.isEmpty ? 'GitHub 直连' : '镜像 ${Uri.parse(p).host}';
    return ModelSource(name, '$p$_repo/$filename');
  }).toList();
}

final Map<String, ModelConfig> modelConfigs = {
  'deoldify_fp32': ModelConfig(
    key: 'deoldify_fp32',
    filename: 'deoldify_fp32.onnx',
    label: 'DeOldify (fp32, 243MB, 高质量)',
    sources: [
      ..._githubSources('deoldify_fp32.onnx'),
      const ModelSource(
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
      ..._githubSources('deoldify_int8.onnx'),
      const ModelSource(
        'GitHub MartinDelophy (备用)',
        'https://github.com/MartinDelophy/deoldify-onnx-web/releases/download/v1.0.0/deoldify.quant.onnx',
      ),
    ],
  ),
  'ddcolor': ModelConfig(
    key: 'ddcolor',
    filename: 'ddcolor_fp32.onnx',
    label: 'DDColor (fp32, 934MB, 漫画风)',
    sources: [
      ..._githubSources('ddcolor_fp32.onnx'),
      const ModelSource(
        'HuggingFace facefusion (备用)',
        'https://huggingface.co/facefusion/models-3.0.0/resolve/main/ddcolor.onnx',
      ),
    ],
  ),
};
