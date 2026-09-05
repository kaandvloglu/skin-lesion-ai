package com.skinlesion.skin_lesion_backend.service;

import com.skinlesion.skin_lesion_backend.dto.PredictionResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
public class PredictionService {

    private final RestClient restClient;
    private final String aiServiceUrl;

    public PredictionService(
            RestClient restClient,
            @Value("${ai.service.url}") String aiServiceUrl) {

        this.restClient = restClient;
        this.aiServiceUrl = aiServiceUrl;
    }

    public PredictionResponse predict(
            MultipartFile clinicalImage,
            MultipartFile dermoscopicImage,
            int age,
            String sex,
            int skinTone,
            String site) {

        try {
            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

            body.add("clinical_image", createFileResource(clinicalImage));
            body.add("dermoscopic_image", createFileResource(dermoscopicImage));
            body.add("age", age);
            body.add("sex", sex);
            body.add("skin_tone", skinTone);
            body.add("site", site);

            return restClient.post()
                    .uri(aiServiceUrl + "/predict")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .retrieve()
                    .body(PredictionResponse.class);

        } catch (IOException e) {
            throw new RuntimeException("Could not read uploaded images", e);
        }
    }

    private ByteArrayResource createFileResource(MultipartFile file)
            throws IOException {

        return new ByteArrayResource(file.getBytes()) {
            @Override
            public String getFilename() {
                return file.getOriginalFilename();
            }
        };
    }
}