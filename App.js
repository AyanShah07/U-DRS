import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  StatusBar,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { analyzeImage, checkHealth } from './services/api';
import { SEVERITY_COLORS, URGENCY_COLORS } from './config/api';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkAPIHealth();
    requestPermissions();
  }, []);

  const checkAPIHealth = async () => {
    const result = await checkHealth();
    setApiStatus(result.success ? 'connected' : 'disconnected');
    if (!result.success) {
      Alert.alert(
        'API Connection Error',
        'Cannot connect to U-DRS backend. Please ensure the server is running on http://127.0.0.1:8080'
      );
    }
  };

  const requestPermissions = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Camera roll permission is needed to select images');
    }
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setAnalysisResult(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Camera permission is needed to take photos');
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedImage(result.assets[0].uri);
      setAnalysisResult(null);
    }
  };

  const analyzeSelectedImage = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or capture an image first');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    const result = await analyzeImage(selectedImage);

    setIsAnalyzing(false);

    if (result.success) {
      setAnalysisResult(result.data);
    } else {
      Alert.alert('Analysis Failed', result.error);
    }
  };

  const renderStatusIndicator = () => {
    const statusColor = apiStatus === 'connected' ? '#4CAF50' : '#F44336';
    const statusText = apiStatus === 'connected' ? 'Connected' : 'Disconnected';

    return (
      <View style={[styles.statusBadge, { backgroundColor: statusColor }]}>
        <Text style={styles.statusText}>{statusText}</Text>
      </View>
    );
  };

  const renderAnalysisResults = () => {
    if (!analysisResult) return null;

    const { status, measurements, severity, cost_urgency } = analysisResult;

    if (status === 'no_damage') {
      return (
        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>✅ No Damage Detected</Text>
          <Text style={styles.resultSubtitle}>
            The analyzed image shows no signs of damage
          </Text>
        </View>
      );
    }

    const summary = measurements?.summary || {};
    const severityClass = severity?.class || 'unknown';
    const severityScore = severity?.score || 0;
    const costPrediction = cost_urgency?.cost_prediction || {};
    const urgency = cost_urgency?.urgency || 'unknown';

    return (
      <ScrollView style={styles.resultsContainer}>
        {/* Severity Card */}
        <View style={[styles.resultCard, { borderLeftColor: SEVERITY_COLORS[severityClass], borderLeftWidth: 5 }]}>
          <Text style={styles.resultTitle}>🛡️ Severity Assessment</Text>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Class:</Text>
            <Text style={[styles.resultValue, { color: SEVERITY_COLORS[severityClass] }]}>
              {severityClass.toUpperCase()}
            </Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Score:</Text>
            <Text style={styles.resultValue}>{severityScore.toFixed(1)} / 100</Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Confidence:</Text>
            <Text style={styles.resultValue}>{((severity?.confidence || 0) * 100).toFixed(0)}%</Text>
          </View>
        </View>

        {/* 2D Measurements Card */}
        <View style={styles.resultCard}>
          <Text style={styles.resultTitle}>📏 Measurements (2D)</Text>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Damage Area:</Text>
            <Text style={styles.resultValue}>{summary.damage_area_mm2?.toFixed(1) || 0} mm²</Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Crack Length:</Text>
            <Text style={styles.resultValue}>{summary.crack_length_mm?.toFixed(1) || 0} mm</Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Crack Width (Mean):</Text>
            <Text style={styles.resultValue}>{summary.crack_width_mean_mm?.toFixed(2) || 0} mm</Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Crack Width (Max):</Text>
            <Text style={styles.resultValue}>{summary.crack_width_max_mm?.toFixed(2) || 0} mm</Text>
          </View>
        </View>

        {/* 3D Measurements Card (if available) */}
        {summary.max_depth_mm && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>📐 Measurements (3D)</Text>
            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Max Depth:</Text>
              <Text style={styles.resultValue}>{summary.max_depth_mm?.toFixed(2) || 0} mm</Text>
            </View>
            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Mean Depth:</Text>
              <Text style={styles.resultValue}>{summary.mean_depth_mm?.toFixed(2) || 0} mm</Text>
            </View>
            {summary.volume_mm3 && (
              <View style={styles.resultRow}>
                <Text style={styles.resultLabel}>Volume:</Text>
                <Text style={styles.resultValue}>{summary.volume_mm3?.toFixed(1) || 0} mm³</Text>
              </View>
            )}
          </View>
        )}

        {/* Cost & Urgency Card */}
        <View style={[styles.resultCard, { borderLeftColor: URGENCY_COLORS[urgency], borderLeftWidth: 5 }]}>
          <Text style={styles.resultTitle}>💰 Cost & Urgency</Text>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Estimated Cost:</Text>
            <Text style={[styles.resultValue, styles.costText]}>
              ${costPrediction.estimated_cost_usd?.toFixed(2) || 0}
            </Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Cost Range:</Text>
            <Text style={styles.resultValue}>
              ${costPrediction.lower_bound_usd?.toFixed(2) || 0} - ${costPrediction.upper_bound_usd?.toFixed(2) || 0}
            </Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Urgency:</Text>
            <Text style={[styles.resultValue, { color: URGENCY_COLORS[urgency] }]}>
              {urgency.toUpperCase()}
            </Text>
          </View>
          <View style={styles.resultRow}>
            <Text style={styles.resultLabel}>Timeline:</Text>
            <Text style={styles.resultValue}>{cost_urgency?.recommended_timeline || 'N/A'}</Text>
          </View>
          <Text style={styles.urgencyDescription}>{cost_urgency?.urgency_description || ''}</Text>
        </View>
      </ScrollView>
    );
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />

      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>U-DRS Mobile</Text>
        <Text style={styles.headerSubtitle}>Damage Reconstruction System</Text>
        {renderStatusIndicator()}
      </View>

      {/* Main Content */}
      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        {/* Action Buttons */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.primaryButton]}
            onPress={takePhoto}
            disabled={apiStatus !== 'connected'}
          >
            <Text style={styles.buttonText}>📷 Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.secondaryButton]}
            onPress={pickImage}
            disabled={apiStatus !== 'connected'}
          >
            <Text style={styles.buttonText}>🖼️ Choose Photo</Text>
          </TouchableOpacity>
        </View>

        {/* Selected Image Preview */}
        {selectedImage && (
          <View style={styles.imageContainer}>
            <Text style={styles.sectionTitle}>Selected Image:</Text>
            <Image source={{ uri: selectedImage }} style={styles.previewImage} />

            <TouchableOpacity
              style={[styles.button, styles.analyzeButton]}
              onPress={analyzeSelectedImage}
              disabled={isAnalyzing || apiStatus !== 'connected'}
            >
              {isAnalyzing ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.buttonText}>🔍 Analyze Damage</Text>
              )}
            </TouchableOpacity>
          </View>
        )}

        {/* Loading Indicator */}
        {isAnalyzing && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color="#2196F3" />
            <Text style={styles.loadingText}>Analyzing damage...</Text>
            <Text style={styles.loadingSubtext}>This may take 30-60 seconds</Text>
          </View>
        )}

        {/* Analysis Results */}
        {renderAnalysisResults()}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2196F3',
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: '#fff',
    opacity: 0.9,
    marginBottom: 10,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
    marginTop: 5,
  },
  statusText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  button: {
    flex: 1,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginHorizontal: 5,
  },
  primaryButton: {
    backgroundColor: '#2196F3',
  },
  secondaryButton: {
    backgroundColor: '#4CAF50',
  },
  analyzeButton: {
    backgroundColor: '#FF9800',
    marginTop: 15,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#333',
    marginBottom: 10,
  },
  imageContainer: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  previewImage: {
    width: '100%',
    height: 300,
    borderRadius: 10,
    resizeMode: 'cover',
  },
  loadingContainer: {
    padding: 30,
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#333',
    fontWeight: '600',
  },
  loadingSubtext: {
    marginTop: 5,
    fontSize: 14,
    color: '#666',
  },
  resultsContainer: {
    flex: 1,
  },
  resultCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#333',
    marginBottom: 12,
  },
  resultSubtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 5,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  resultLabel: {
    fontSize: 14,
    color: '#666',
    flex: 1,
  },
  resultValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    flex: 1,
    textAlign: 'right',
  },
  costText: {
    color: '#4CAF50',
    fontSize: 16,
  },
  urgencyDescription: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
    fontStyle: 'italic',
  },
});
