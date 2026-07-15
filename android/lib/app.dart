import 'package:flutter/material.dart';
import 'home_page.dart';

class AiColorizeApp extends StatelessWidget {
  const AiColorizeApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AiColorize',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.teal,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const HomePage(),
    );
  }
}
