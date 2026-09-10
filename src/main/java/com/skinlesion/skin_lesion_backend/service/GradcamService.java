package com.skinlesion.skin_lesion_backend.service;

import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.multipart.MultipartFile;

@Service
public class GradcamService {

    private final RestClient restClient;

    @Value("${gradcam.service.url}")
    private String gradcamServiceUrl;

    public GradcamService(RestClient restClient) {
        this.restClient = restClient;
    }

    public Map<String, String> generateGradcam(
            MultipartFile clinicalImage,
            MultipartFile dermoscopicImage,
            Integer age,
            String sex,
            Integer skinTone,
            String site) throws Exception {

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();

        body.add(
                "clinical_image",
                new ByteArrayResource(clinicalImage.getBytes()) {
                    @Override
                    public String getFilename() {
                        return clinicalImage.getOriginalFilename();
                    }
                });

        body.add(
                "dermoscopic_image",
                new ByteArrayResource(dermoscopicImage.getBytes()) {
                    @Override
                    public String getFilename() {
                        return dermoscopicImage.getOriginalFilename();
                    }
                });

        body.add("age", age.toString());
        body.add("sex", sex);
        body.add("skin_tone", skinTone.toString());
        body.add("site", site);

        return restClient
                .post()
                .uri(gradcamServiceUrl + "/gradcam")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(new ParameterizedTypeReference<Map<String, String>>() {});
    }
}