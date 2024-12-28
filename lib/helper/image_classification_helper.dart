// ignore_for_file: avoid_print, await_only_futures, unused_element

/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'dart:developer';
import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import 'isolate_inference.dart';

class ImageClassificationHelper {
  // static const modelPath = 'assets/models/mobilenet_quant.tflite';
  static const modelUrl =
      'https://raw.githubusercontent.com/SafwanSadaan/image_classification_mobilenet/refs/heads/main/assets/models/ArASL-BestModel_full_integer.tflite';
  static const labelsUrl =
      'https://raw.githubusercontent.com/SafwanSadaan/image_classification_mobilenet/refs/heads/main/assets/models/labels1.txt';
  static const modelPath = 'assets/models/ArASL-BestModel_full_integer.tflite';
  static const labelsPath = 'assets/models/labels1.txt';
  // static const modelPath = 'assets/models/RecommendedPlantCareModel.tflite';
  // static const labelsPath = 'assets/models/labels.txt';

  late final Interpreter interpreter;
  late final List<String> labels;
  late final IsolateInference isolateInference;
  late Tensor inputTensor;
  late Tensor outputTensor;

  // Load model
  Future<void> _loadModel() async {
    final options = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    // Use GPU Delegate
    // doesn't work on emulator
    // if (Platform.isAndroid) {
    //   options.addDelegate(GpuDelegateV2());
    // }

    // Use Metal Delegate
    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    // Load model from assets
    interpreter = await Interpreter.fromAsset(modelPath, options: options);
    // Get tensor input shape [1, 224, 224, 3]
    inputTensor = interpreter.getInputTensors().first;
    // Get tensor output shape [1, 1001]
    outputTensor = interpreter.getOutputTensors().first;

    log('Interpreter loaded successfully');
  }

  // Load labels from assets
  Future<void> _loadLabels() async {
    final labelTxt = await rootBundle.loadString(labelsPath);
    labels = labelTxt.split('\n');
  }

  Future<void> initHelper() async {
    _loadLabels();
    _loadModel();
    isolateInference = IsolateInference();
    await isolateInference.start();
  }

  Future<Map<String, double>> _inference(InferenceModel inferenceModel) async {
    ReceivePort responsePort = ReceivePort();
    isolateInference.sendPort
        .send(inferenceModel..responsePort = responsePort.sendPort);
    // get inference result.
    var results = await responsePort.first;
    return results;
  }

  // inference camera frame
  Future<Map<String, double>> inferenceCameraFrame(
      CameraImage cameraImage) async {
    var isolateModel = InferenceModel(cameraImage, null, interpreter.address,
        labels, inputTensor.shape, outputTensor.shape);
    return _inference(isolateModel);
  }

  // inference still image
  Future<Map<String, double>> inferenceImage(Image image) async {
    var isolateModel = InferenceModel(null, image, interpreter.address, labels,
        inputTensor.shape, outputTensor.shape);
    return _inference(isolateModel);
  }

  Future<void> close() async {
    isolateInference.close();
  }

// Load model from URL
  Future<void> _loadModelUrl() async {
    final options = InterpreterOptions();

    // Use XNNPACK Delegate
    if (Platform.isAndroid) {
      options.addDelegate(XNNPackDelegate());
    }

    // Use Metal Delegate for iOS
    if (Platform.isIOS) {
      options.addDelegate(GpuDelegate());
    }

    // Download model and labels
    final modelPath = await _downloadFile(modelUrl, 'model.tflite');
    final labelsPath = await _downloadFile(labelsUrl, 'labels.txt');

    // Load model from downloaded path
    interpreter = await Interpreter.fromFile(File(modelPath), options: options);

    // Load labels if available
    labels = await File(labelsPath).readAsLines();

    // Get tensor input and output shapes
    inputTensor = interpreter.getInputTensors().first;
    outputTensor = interpreter.getOutputTensors().first;

    log('Interpreter and labels loaded successfully');
  }

  // Helper function to download a file and save it locally
  Future<String> _downloadFile(String url, String filename) async {
    final response = await http.get(Uri.parse(url));
    if (response.statusCode == 200) {
      final directory = await getApplicationDocumentsDirectory();
      final filePath = '${directory.path}/$filename';
      final file = File(filePath);
      await file.writeAsBytes(response.bodyBytes);
      log('$filename downloaded successfully');
      return filePath;
    } else {
      throw Exception('Failed to download $filename from $url');
    }
  }
}
