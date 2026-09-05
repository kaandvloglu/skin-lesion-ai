package com.skinlesion.skin_lesion_backend.dto;

import java.util.Map;

public class PredictionResponse {

    private String prediction;
    private double confidence;
    private Map<String, Double> scores;

    public PredictionResponse() {
    }

    public String getPrediction() {
        return prediction;
    }

    public void setPrediction(String prediction) {
        this.prediction = prediction;
    }

    public double getConfidence() {
        return confidence;
    }

    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }

    public Map<String, Double> getScores() {
        return scores;
    }

    public void setScores(Map<String, Double> scores) {
        this.scores = scores;
    }
}