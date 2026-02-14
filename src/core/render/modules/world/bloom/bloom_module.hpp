#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/modules/world/world_module.hpp"

class Framework;
class FrameworkContext;
class WorldPipeline;
struct WorldModuleContext;

struct BloomModuleContext;

struct BloomDownsamplePushConstant {
    float srcTexelSizeX;
    float srcTexelSizeY;
    float threshold;
    float softKnee;
    int applyThreshold;
    float padding0;
    float padding1;
    float padding2;
};

struct BloomUpsamplePushConstant {
    float srcTexelSizeX;
    float srcTexelSizeY;
    float bloomRadius;
    float padding;
};

struct BloomCompositePushConstant {
    float intensity;
    float padding0;
    float padding1;
    float padding2;
};

class BloomModule : public WorldModule, public SharedObject<BloomModule> {
    friend BloomModuleContext;

  public:
    constexpr static std::string_view NAME = "render_pipeline.module.bloom.name";
    constexpr static uint32_t inputImageNum = 1;
    constexpr static uint32_t outputImageNum = 1;
    constexpr static uint32_t MAX_MIP_LEVELS = 6;

    BloomModule();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline);

    bool setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                std::vector<VkFormat> &formats,
                                uint32_t frameIndex) override;
    bool setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                 std::vector<VkFormat> &formats,
                                 uint32_t frameIndex) override;

    void setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) override;

    void build() override;

    std::vector<std::shared_ptr<WorldModuleContext>> &contexts() override;

    void
    bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index) override;

    void preClose() override;

  private:
    void initDescriptorTables();
    void initMipChain();
    void initPipelines();

  private:
    // input
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrInputImages_;

    // output
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> hdrOutputImages_;

    // bloom mip chain (per frame × per mip level)
    // mipImages_[frameIndex][mipLevel]
    std::vector<std::vector<std::shared_ptr<vk::DeviceLocalImage>>> mipImages_;
    uint32_t mipCount_ = 0;

    // descriptor tables: one per frame per pass phase
    // downsampleDescTables_[frameIndex][mipLevel]
    std::vector<std::vector<std::shared_ptr<vk::DescriptorTable>>> downsampleDescTables_;
    // upsampleDescTables_[frameIndex][mipLevel]
    std::vector<std::vector<std::shared_ptr<vk::DescriptorTable>>> upsampleDescTables_;
    // compositeDescTables_[frameIndex]
    std::vector<std::shared_ptr<vk::DescriptorTable>> compositeDescTables_;

    std::vector<std::shared_ptr<vk::Sampler>> samplers_;

    // compute pipelines
    std::shared_ptr<vk::Shader> downsampleShader_;
    std::shared_ptr<vk::ComputePipeline> downsamplePipeline_;

    std::shared_ptr<vk::Shader> upsampleShader_;
    std::shared_ptr<vk::ComputePipeline> upsamplePipeline_;

    std::shared_ptr<vk::Shader> compositeShader_;
    std::shared_ptr<vk::ComputePipeline> compositePipeline_;

    // configurable attributes
    float intensity_ = 0.3f;
    float threshold_ = 1.0f;
    float softKnee_ = 0.5f;
    float radius_ = 1.0f;

    uint32_t width_, height_;

    std::vector<std::shared_ptr<WorldModuleContext>> contexts_;
};

struct BloomModuleContext : public WorldModuleContext, SharedObject<BloomModuleContext> {
    std::weak_ptr<BloomModule> bloomModule;
    uint32_t frameIndex;

    // input
    std::shared_ptr<vk::DeviceLocalImage> hdrInputImage;

    // output
    std::shared_ptr<vk::DeviceLocalImage> hdrOutputImage;

    BloomModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                       std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                       std::shared_ptr<BloomModule> bloomModule,
                       uint32_t frameIndex);

    void render() override;
};
