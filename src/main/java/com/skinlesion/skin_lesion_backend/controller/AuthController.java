 package com.skinlesion.skin_lesion_backend.controller;

import com.skinlesion.skin_lesion_backend.dto.LoginRequest;
import com.skinlesion.skin_lesion_backend.dto.RegisterRequest;
import com.skinlesion.skin_lesion_backend.model.User;
import com.skinlesion.skin_lesion_backend.service.AuthService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContext;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.web.context.HttpSessionSecurityContextRepository;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    public ResponseEntity<User> register(@Valid @RequestBody RegisterRequest request) {

        User user = authService.register(
                request.getName(),
                request.getEmail(),
                request.getPassword()
        );

        return ResponseEntity.ok(user);
    }

    @PostMapping("/login")
    public ResponseEntity<User> login(
            @Valid @RequestBody LoginRequest request,
            HttpServletRequest httpRequest) {

        User user = authService.login(
                request.getEmail(),
                request.getPassword()
        );

        UsernamePasswordAuthenticationToken authentication =
                UsernamePasswordAuthenticationToken.authenticated(
                        user.getEmail(),
                        null,
                        List.of(new SimpleGrantedAuthority("ROLE_USER"))
                );

        SecurityContext securityContext =
                SecurityContextHolder.createEmptyContext();

        securityContext.setAuthentication(authentication);
        SecurityContextHolder.setContext(securityContext);

        httpRequest.getSession(true).setAttribute(
                HttpSessionSecurityContextRepository.SPRING_SECURITY_CONTEXT_KEY,
                securityContext
        );

        return ResponseEntity.ok(user);
    }
    @GetMapping("/me")
public ResponseEntity<String> me() {
    String email = SecurityContextHolder
            .getContext()
            .getAuthentication()
            .getName();

    return ResponseEntity.ok("Logged in as: " + email);
}
@PostMapping("/logout")
public ResponseEntity<String> logout(HttpServletRequest request) {

    var session = request.getSession(false);

    if (session != null) {
        session.invalidate();
    }

    SecurityContextHolder.clearContext();

    return ResponseEntity.ok("Logged out successfully");
}
}