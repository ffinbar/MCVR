#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D HDR;

layout(set = 0, binding = 2) readonly buffer ExposureBuffer {
    float exposure;
    float avgLogLum;
    float padding0;
    float padding1;
}
gExposure;

layout(push_constant) uniform PushConstant {
    float log2Min;
    float log2Max;
    float epsilon;
    float lowPercent;
    float highPercent;
    float middleGrey;
    float dt;
    float speedUp;
    float speedDown;
    float minExposure;
    float maxExposure;
    float darkAdaptLimit;
    float saturation;
    float contrast;
} pc;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

// Uchimura (Gran Turismo) tone mapping operator
// P = max brightness, a = contrast, m = linear section start,
// l = linear section length, c = black tightness, b = pedestal (shadow lift)
float uchimura(float x, float P, float a, float m, float l, float c, float b) {
    float l0 = ((P - m) * l) / a;
    float L0 = m - m / a;
    float L1 = m + (1.0 - m) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float w0 = 1.0 - smoothstep(0.0, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0 - w0 - w2;

    float T = m * pow(x / m, c) + b;
    float S = P - (P - S1) * exp(CP * (x - S0));
    float L = m + a * (x - m);

    return T * w0 + L * w1 + S * w2;
}

vec3 uchimuraToneMap(vec3 x) {
    const float P = 1.0;   // max display brightness
    const float a = 1.0;   // contrast
    const float m = 0.22;  // linear section start
    const float l = 0.4;   // linear section length
    const float c = 1.33;  // black tightness
    const float b = 0.0;   // pedestal

    // Luminance-based tonemapping: apply curve to luminance only,
    // then scale color proportionally. This preserves hue and saturation
    // in bright highlights instead of desaturating toward white.
    float Lin = dot(x, vec3(0.2126, 0.7152, 0.0722));
    if (Lin <= 0.0) return vec3(0.0);
    float Lout = uchimura(Lin, P, a, m, l, c, b);
    vec3 mapped = x * (Lout / Lin);

    // Gamut mapping: scale uniformly to fit in [0,1] while perfectly
    // preserving hue ratios. Only reduces brightness, never desaturates.
    float maxC = max(mapped.r, max(mapped.g, mapped.b));
    if (maxC > 1.0) mapped /= maxC;
    return mapped;
}

// Linear → sRGB transfer function.
// Required because the swapchain is R8G8B8A8_UNORM (no auto-conversion).
vec3 linearToSrgb(vec3 linear) {
    vec3 lo = linear * 12.92;
    vec3 hi = 1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), linear));
}

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    vec3 mapped = uchimuraToneMap(expColor);

    // Saturation adjustment: mix between luminance (grey) and color
    float lum = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
    mapped = max(mix(vec3(lum), mapped, pc.saturation), 0.0);

    // Contrast adjustment around perceptual midpoint
    mapped = pow(max(mapped, vec3(0.0)), vec3(pc.contrast));

    // Encode to sRGB for correct display on UNORM swapchain
    fragColor = vec4(linearToSrgb(mapped), 1.0);
}
