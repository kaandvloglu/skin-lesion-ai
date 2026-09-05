package com.skinlesion.skin_lesion_backend.controller;

import com.skinlesion.skin_lesion_backend.dto.PredictionResponse;
import com.skinlesion.skin_lesion_backend.service.PredictionService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    @GetMapping("/test")
    public ResponseEntity<String> test() {
        return ResponseEntity.ok("Prediction API is working");
    }

    @PostMapping("/upload")
    public ResponseEntity<PredictionResponse> uploadImages(
            @RequestParam("clinical_image") MultipartFile clinicalImage,
            @RequestParam("dermoscopic_image") MultipartFile dermoscopicImage,
            @RequestParam("age") int age,
            @RequestParam("sex") String sex,
            @RequestParam("skin_tone") int skinTone,
            @RequestParam("site") String site) {

        PredictionResponse result = predictionService.predict(
                clinicalImage,
                dermoscopicImage,
                age,
                sex,
                skinTone,
                site
        );

        return ResponseEntity.ok(result);
    }
}