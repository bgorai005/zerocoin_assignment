__kernel void gaussian_blur(
    __global const uchar4* input,
    __global uchar4* output,
    const int width,
    const int height)
{
    const int filter[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const int kernel_sum = 16;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;

    int sum_r = 0, sum_g = 0, sum_b = 0;

    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int nx = clamp(x + kx, 0, width - 1);
            int ny = clamp(y + ky, 0, height - 1);
            uchar4 pixel = input[ny * width + nx];
            int k_idx = (ky + 1) * 3 + (kx + 1);
            int weight = filter[k_idx];

            sum_r += pixel.x * weight;
            sum_g += pixel.y * weight;
            sum_b += pixel.z * weight;
        }
    }

    uchar4 result;
    result.x = (uchar)(sum_r / kernel_sum);
    result.y = (uchar)(sum_g / kernel_sum);
    result.z = (uchar)(sum_b / kernel_sum);
    result.w = input[y * width + x].w; 

    output[y * width + x] = result;
}
__kernel void log_tone_map(
    __global const uchar4* input,
    __global uchar4* output,
    const float max_luminance,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int index = y * width + x;
    uchar4 pixel = input[index];

    // Convert to normalized float values [0.0, 1.0]
    float R = (float)pixel.x / 255.0f;
    float G = (float)pixel.y / 255.0f;
    float B = (float)pixel.z / 255.0f;

    // Compute luminance Y
    float Y = 0.2126f * R + 0.7152f * G + 0.0722f * B;

    // Avoid divide by zero
    float Y_out = (Y > 0.0f) ?
        log(1.0f + Y) / log(1.0f + max_luminance) :
        0.0f;

    // Rescale R, G, B while preserving ratios
    float R_out = (Y > 0.0f) ? R * (Y_out / Y) : 0.0f;
    float G_out = (Y > 0.0f) ? G * (Y_out / Y) : 0.0f;
    float B_out = (Y > 0.0f) ? B * (Y_out / Y) : 0.0f;

    // Clamp and convert back to 8-bit
    uchar4 result;
    result.x = (uchar)(clamp(R_out * 255.0f, 0.0f, 255.0f));
    result.y = (uchar)(clamp(G_out * 255.0f, 0.0f, 255.0f));
    result.z = (uchar)(clamp(B_out * 255.0f, 0.0f, 255.0f));
    result.w = pixel.w; // Preserve alpha

    output[index] = result;
}
