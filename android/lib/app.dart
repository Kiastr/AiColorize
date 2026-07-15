import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'home_page.dart';

class AiColorizeApp extends StatefulWidget {
  const AiColorizeApp({super.key});

  @override
  State<AiColorizeApp> createState() => _AiColorizeAppState();
}

class _AiColorizeAppState extends State<AiColorizeApp> {
  // 主题模式：system / light / dark，默认跟随系统
  final ValueNotifier<ThemeMode> _themeMode = ValueNotifier(ThemeMode.system);

  @override
  void initState() {
    super.initState();
    _loadThemeMode();
  }

  // 启动时从 SharedPreferences 恢复用户选择
  Future<void> _loadThemeMode() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('theme_mode');
    if (saved == 'light') {
      _themeMode.value = ThemeMode.light;
    } else if (saved == 'dark') {
      _themeMode.value = ThemeMode.dark;
    } else {
      _themeMode.value = ThemeMode.system;
    }
  }

  @override
  void dispose() {
    _themeMode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // themeMode 变化时实时重渲 MaterialApp，无需重启
    return ValueListenableBuilder<ThemeMode>(
      valueListenable: _themeMode,
      builder: (context, mode, _) {
        return MaterialApp(
          title: 'AiColorize',
          debugShowCheckedModeBanner: false,
          themeMode: mode,
          theme: ThemeData.light().copyWith(
            // fromSeed 按亮度生成主题色：浅色用常规 teal，深色自动取浅色调跟随系统
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.teal,
              brightness: Brightness.light,
            ),
            scaffoldBackgroundColor: Colors.grey[50],
            cardColor: Colors.white,
            visualDensity: VisualDensity.adaptivePlatformDensity,
          ),
          // 暗色主题：背景/卡片转深色，主题色(teal)自动取浅色调跟随系统
          darkTheme: ThemeData.dark().copyWith(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.teal,
              brightness: Brightness.dark,
            ),
            scaffoldBackgroundColor: Colors.grey[900],
            cardColor: Colors.grey[850],
            visualDensity: VisualDensity.adaptivePlatformDensity,
          ),
          home: HomePage(themeMode: _themeMode),
        );
      },
    );
  }
}
