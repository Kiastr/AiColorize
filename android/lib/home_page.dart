import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'models/model_config.dart';
import 'services/colorize_service.dart';
import 'services/model_downloader.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  // 配置状态
  String _mode = 'single'; // 'single' | 'batch'
  String _engine = 'ddcolor'; // 'ddcolor' | 'deoldify'
  bool _useNnapi = true;

  // 模型状态
  String? _modelPath;
  bool _downloading = false;
  double _downloadProgress = 0;

  // 输入/输出
  String? _inputPath; // 单图
  String? _inputDir; // 批量输入目录
  String? _outputDir; // 批量输出目录

  // 运行状态
  bool _processing = false;
  int _processed = 0;
  int _total = 0;
  String? _resultPath; // 单图结果
  String _log = '';

  final ModelDownloader _downloader = ModelDownloader();

  @override
  void initState() {
    super.initState();
    _requestPermissions();
  }

  Future<void> _requestPermissions() async {
    await Permission.storage.request();
    if (await Permission.photos.request().isDenied) {
      // 低版本用 storage 已覆盖
    }
  }

  ModelConfig get _currentConfig => modelConfigs[_engine]!;

  Future<String> _modelsDir() async {
    // 用外部存储目录（/storage/emulated/0/Android/data/<pkg>/files/models），
    // 便于用户用文件管理器手动放入/查看 ONNX 模型，无需 root
    final dir = await getExternalStorageDirectory();
    final models = Directory('${dir!.path}/models');
    if (!await models.exists()) {
      await models.create(recursive: true);
    }
    return models.path;
  }

  Future<void> _pickLocalModel() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['onnx'],
    );
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _modelPath = result.files.first.path;
        _log = '已选择本地模型: $_modelPath';
      });
    }
  }

  Future<void> _downloadModel() async {
    if (_downloading) return;
    setState(() {
      _downloading = true;
      _downloadProgress = 0;
      _log = '开始下载模型...';
    });
    final dir = await _modelsDir();
    final savePath = '$dir/${_currentConfig.filename}';
    try {
      await _downloader.download(
        config: _currentConfig,
        savePath: savePath,
        onProgress: (received, total) {
          if (mounted) {
            setState(() {
              _downloadProgress = total > 0 ? received / total : 0;
            });
          }
        },
      );
      setState(() {
        _modelPath = savePath;
        _log = '模型下载完成: $savePath';
      });
    } catch (e) {
      setState(() {
        _log = '下载失败: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _downloading = false;
        });
      }
    }
  }

  Future<void> _pickInput() async {
    if (_mode == 'single') {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
      );
      if (result != null && result.files.isNotEmpty) {
        setState(() {
          _inputPath = result.files.first.path;
          _resultPath = null;
        });
      }
    } else {
      final dir = await FilePicker.platform.getDirectoryPath();
      if (dir != null) {
        setState(() {
          _inputDir = dir;
        });
      }
    }
  }

  Future<void> _pickOutputDir() async {
    final dir = await FilePicker.platform.getDirectoryPath();
    if (dir != null) {
      setState(() {
        _outputDir = dir;
      });
    }
  }

  Future<void> _start() async {
    if (_modelPath == null || !File(_modelPath!).existsSync()) {
      _setLog('请先下载或选择模型');
      return;
    }
    if (_mode == 'single') {
      if (_inputPath == null) {
        _setLog('请先选择输入图片');
        return;
      }
      await _runSingle();
    } else {
      if (_inputDir == null || _outputDir == null) {
        _setLog('请先选择输入文件夹与输出目录');
        return;
      }
      await _runBatch();
    }
  }

  Future<void> _runSingle() async {
    setState(() {
      _processing = true;
      _log = '正在上色...';
    });
    try {
      final out = await ColorizeService.colorize(
        inputPath: _inputPath!,
        modelPath: _modelPath!,
        type: _engine,
        useNnapi: _useNnapi,
      );
      setState(() {
        _resultPath = out;
        _log = '上色完成，已保存到: $out';
      });
    } catch (e) {
      _setLog('上色失败: $e');
    } finally {
      if (mounted) setState(() => _processing = false);
    }
  }

  Future<void> _runBatch() async {
    final files = Directory(_inputDir!)
        .listSync()
        .whereType<File>()
        .where((f) {
          final ext = f.path.toLowerCase();
          return ext.endsWith('.png') ||
              ext.endsWith('.jpg') ||
              ext.endsWith('.jpeg') ||
              ext.endsWith('.bmp') ||
              ext.endsWith('.webp');
        })
        .toList();

    if (files.isEmpty) {
      _setLog('文件夹内没有支持的图片文件');
      return;
    }

    // 危险操作确认
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('批量上色'),
        content: Text('将处理 ${files.length} 张图片，输出到所选目录。是否继续？'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('取消')),
          TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              child: const Text('继续')),
        ],
      ),
    );
    if (confirm != true) return;

    setState(() {
      _processing = true;
      _total = files.length;
      _processed = 0;
    });

    int failed = 0;
    for (final f in files) {
      final name = f.uri.pathSegments.last;
      final outName = 'colorized_$name';
      final outPath = '$_outputDir/$outName';
      try {
        await ColorizeService.colorize(
          inputPath: f.path,
          modelPath: _modelPath!,
          type: _engine,
          useNnapi: _useNnapi,
          outputPath: outPath,
        );
      } catch (e) {
        failed++;
      }
      if (mounted) {
        setState(() {
          _processed++;
          _log = '处理中: $_processed / $_total  (失败 $failed)';
        });
      }
    }
    if (mounted) {
      setState(() => _processing = false);
      _setLog('批量完成：$_processed 张，失败 $failed 张。结果目录: $_outputDir');
    }
  }

  void _setLog(String msg) {
    if (mounted) setState(() => _log = msg);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AiColorize · 漫画 AI 上色'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildModeSection(),
            const SizedBox(height: 16),
            _buildConfigSection(),
            const SizedBox(height: 16),
            _buildModelSection(),
            const SizedBox(height: 16),
            _buildInputSection(),
            const SizedBox(height: 16),
            _buildActionSection(),
            const SizedBox(height: 16),
            _buildLogSection(),
            if (_mode == 'single' && _resultPath != null) ...[
              const SizedBox(height: 16),
              _buildPreviewSection(),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildModeSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('处理模式', style: TextStyle(fontWeight: FontWeight.bold)),
            Row(
              children: [
                Expanded(
                  child: RadioListTile<String>(
                    title: const Text('单图处理'),
                    value: 'single',
                    groupValue: _mode,
                    onChanged: (v) => setState(() => _mode = v!),
                  ),
                ),
                Expanded(
                  child: RadioListTile<String>(
                    title: const Text('批量处理'),
                    value: 'batch',
                    groupValue: _mode,
                    onChanged: (v) => setState(() => _mode = v!),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildConfigSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('模型引擎', style: TextStyle(fontWeight: FontWeight.bold)),
            DropdownButton<String>(
              value: _engine,
              isExpanded: true,
              items: modelConfigs.values
                  .map((c) => DropdownMenuItem(
                        value: c.key,
                        child: Text(c.label),
                      ))
                  .toList(),
              onChanged: (v) => setState(() => _engine = v!),
            ),
            const SizedBox(height: 8),
            const Text('运行设备', style: TextStyle(fontWeight: FontWeight.bold)),
            DropdownButton<bool>(
              value: _useNnapi,
              isExpanded: true,
              items: const [
                DropdownMenuItem(value: true, child: Text('NNAPI (GPU/NPU 加速)')),
                DropdownMenuItem(value: false, child: Text('CPU')),
              ],
              onChanged: (v) => setState(() => _useNnapi = v!),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('AI 模型', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _downloading ? null : _downloadModel,
                    icon: const Icon(Icons.download),
                    label: Text(_downloading ? '下载中...' : '自动下载模型'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _downloading ? null : _pickLocalModel,
                    icon: const Icon(Icons.folder_open),
                    label: const Text('选择本地模型'),
                  ),
                ),
              ],
            ),
            if (_downloading) ...[
              const SizedBox(height: 8),
              LinearProgressIndicator(value: _downloadProgress),
              Text('${(_downloadProgress * 100).toStringAsFixed(0)}%'),
            ],
            const SizedBox(height: 8),
            Text(
              _modelPath == null ? '尚未准备模型' : '模型: $_modelPath',
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInputSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('输入 / 输出', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _processing ? null : _pickInput,
                    child: Text(_mode == 'single' ? '选择图片' : '选择输入文件夹'),
                  ),
                ),
              ],
            ),
            if (_mode == 'batch') ...[
              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: _processing ? null : _pickOutputDir,
                child: const Text('选择输出目录'),
              ),
              Text(_outputDir == null ? '未选择输出目录' : '输出: $_outputDir',
                  style: const TextStyle(fontSize: 12, color: Colors.grey)),
            ],
            const SizedBox(height: 8),
            Text(
              _mode == 'single'
                  ? (_inputPath == null ? '未选择图片' : '输入: $_inputPath')
                  : (_inputDir == null ? '未选择文件夹' : '输入: $_inputDir'),
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionSection() {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          height: 50,
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(primary: Colors.teal),
            onPressed: _processing ? null : _start,
            child: _processing
                ? const CircularProgressIndicator(color: Colors.white)
                : const Text('一键开始上色',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
          ),
        ),
        if (_processing && _total > 0) ...[
          const SizedBox(height: 8),
          LinearProgressIndicator(value: _total > 0 ? _processed / _total : 0),
          Text('批量进度: $_processed / $_total'),
        ],
      ],
    );
  }

  Widget _buildLogSection() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(_log.isEmpty ? '等待操作...' : _log,
          style: const TextStyle(fontSize: 13)),
    );
  }

  Widget _buildPreviewSection() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('结果预览', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: Column(
                    children: [
                      const Text('原图', style: TextStyle(fontSize: 12)),
                      Image.file(File(_inputPath!), height: 200, fit: BoxFit.contain),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Column(
                    children: [
                      const Text('上色后', style: TextStyle(fontSize: 12)),
                      Image.file(File(_resultPath!), height: 200, fit: BoxFit.contain),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
