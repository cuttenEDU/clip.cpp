#ifndef CLIP_H
#define CLIP_H

#include <cstdint>

struct clip_ctx;

#ifdef __cplusplus
extern "C" {
    #endif

    struct clip_text_hparams {
        int32_t n_vocab;
        int32_t num_positions;
        int32_t hidden_size;
        int32_t n_intermediate;
        int32_t projection_dim;
        int32_t n_head;
        int32_t n_layer;
        float eps;
    };

    struct clip_vision_hparams {
        int32_t image_size;
        int32_t patch_size;
        int32_t hidden_size;
        int32_t n_intermediate;
        int32_t projection_dim;
        int32_t n_head;
        int32_t n_layer;
        float eps;
    };

    typedef int32_t clip_vocab_id;

    struct clip_tokens {
        clip_vocab_id *data;
        size_t size;
    };

    clip_ctx *clip_model_load(const char *fname, int verbosity);

    void clip_free(const clip_ctx *ctx);

    clip_text_hparams *clip_get_text_hparams(clip_ctx *ctx);

    clip_vision_hparams *clip_get_vision_hparams(clip_ctx *ctx);

    // RGB uint8 image
    struct clip_image_u8 {
        int32_t nx;
        int32_t ny;
        uint8_t *data;
        size_t size;
    };

    // RGB float32 image (NHWC)
    // Memory layout: RGBRGBRGB...
    struct clip_image_f32 {
        int32_t nx;
        int32_t ny;
        float *data;
        size_t size;
    };

    struct clip_image_u8_batch {
        clip_image_u8 *data;
        size_t size;
    };

    struct clip_image_f32_batch {
        clip_image_f32 *data;
        size_t size;
    };

    bool clip_tokenize(const clip_ctx *ctx, const char *text, clip_tokens *tokens);

    clip_image_u8 *clip_image_u8_make();

    clip_image_f32 *clip_image_f32_make();

    void clip_image_u8_clean(clip_image_u8 *img);

    void clip_image_f32_clean(clip_image_f32 *res);

    void clip_image_u8_free(clip_image_u8 *img);

    void clip_image_f32_free(clip_image_f32 *res);

    bool clip_image_load_from_file(const char *fname, clip_image_u8 *img);

    bool clip_image_preprocess(const clip_ctx *ctx, const clip_image_u8 *img, clip_image_f32 *res);

    bool clip_text_encode(const clip_ctx *ctx, int n_threads, const clip_tokens *tokens, float *vec, bool normalize);

    bool clip_image_encode(const clip_ctx *ctx, int n_threads, clip_image_f32 *img, float *vec, bool normalize);

    void clip_image_batch_preprocess(
        const clip_ctx *ctx, int n_threads, const clip_image_u8_batch *img_inputs, clip_image_f32_batch *imgs_resized
    );

    bool clip_image_batch_encode(
        const clip_ctx *ctx, int n_threads, const clip_image_f32_batch *imgs, float *vec, bool normalize
    );

    // bool image_normalize(const clip_image_u8 *img, clip_image_f32 *res);

    bool clip_compare_text_and_image(
        const clip_ctx *ctx, int n_threads, const char *text, const clip_image_u8 *image, float *score
    );

    float clip_similarity_score(const float *vec1, const float *vec2, int vec_dim);

    bool softmax_with_sorting(float *arr, int length, float *sorted_scores, int *indices);

    bool clip_zero_shot_label_image(
        clip_ctx *ctx, int n_threads, const clip_image_u8 *input_img,
        const char **labels, size_t n_labels, float *scores, int *indices
    );

    bool clip_model_quantize(const char *fname_inp, const char *fname_out, int itype);

    #ifdef __cplusplus
}
#endif

#endif // CLIP_H
