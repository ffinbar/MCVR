#include "core/render/modules/world/bloom/bloom_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

BloomModule::BloomModule() {}

void BloomModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrInputImages_.resize(size);
    hdrOutputImages_.resize(size);
}

bool BloomModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    if (images.size() == 0) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        hdrInputImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrInputImages_[frameIndex] = images[0];
    }

    return true;
}

bool BloomModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                          std::vector<VkFormat> &formats,
                                          uint32_t frameIndex) {
    if (images.size() == 0 || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    hdrOutputImages_[frameIndex] = images[0];

    return true;
}

void BloomModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        if (attributeKVs[2 * i] == "render_pipeline.module.bloom.attribute.intensity") {
            intensity_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.bloom.attribute.threshold") {
            threshold_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.bloom.attribute.soft_knee") {
            softKnee_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.bloom.attribute.radius") {
            radius_ = std::stof(attributeKVs[2 * i + 1]);
        }
    }
}

void BloomModule::build() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    // Compute mip count based on resolution
    uint32_t minDim = std::min(width_, height_);
    mipCount_ = 0;
    uint32_t dim = minDim;
    while (dim > 4 && mipCount_ < MAX_MIP_LEVELS) {
        dim /= 2;
        mipCount_++;
    }
    if (mipCount_ == 0) mipCount_ = 1;

    initMipChain();
    initDescriptorTables();
    initPipelines();

    contexts_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        auto worldPipeline = worldPipeline_.lock();
        contexts_[i] = BloomModuleContext::create(
            framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this(), i);
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &BloomModule::contexts() {
    return contexts_;
}

void BloomModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                              std::shared_ptr<vk::DeviceLocalImage> image,
                              int index) {}

void BloomModule::preClose() {}

void BloomModule::initMipChain() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    mipImages_.resize(size);

    for (uint32_t f = 0; f < size; f++) {
        mipImages_[f].resize(mipCount_);
        uint32_t w = width_;
        uint32_t h = height_;

        for (uint32_t m = 0; m < mipCount_; m++) {
            w = std::max(w / 2, 1u);
            h = std::max(h / 2, 1u);
            mipImages_[f][m] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, w, h, 1,
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }
}

void BloomModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    downsampleDescTables_.resize(size);
    upsampleDescTables_.resize(size);
    compositeDescTables_.resize(size);
    samplers_.resize(size);

    for (uint32_t f = 0; f < size; f++) {
        samplers_[f] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR,
                                           VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

        // Downsample descriptor tables (one per mip level)
        downsampleDescTables_[f].resize(mipCount_);
        for (uint32_t m = 0; m < mipCount_; m++) {
            downsampleDescTables_[f][m] =
                vk::DescriptorTableBuilder{}
                    .beginDescriptorLayoutSet()
                    .beginDescriptorLayoutSetBinding()
                    .defineDescriptorLayoutSetBinding({
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .defineDescriptorLayoutSetBinding({
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .endDescriptorLayoutSetBinding()
                    .endDescriptorLayoutSet()
                    .definePushConstant(VkPushConstantRange{
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .offset = 0,
                        .size = sizeof(BloomDownsamplePushConstant),
                    })
                    .build(framework->device());

            // Source: previous mip (or HDR input for first pass)
            auto srcImage = (m == 0) ? hdrInputImages_[f] : mipImages_[f][m - 1];
            downsampleDescTables_[f][m]->bindSamplerImageForShader(samplers_[f], srcImage, 0, 0);
            downsampleDescTables_[f][m]->bindImage(mipImages_[f][m], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        }

        // Upsample descriptor tables (one per mip level, from bottom up)
        upsampleDescTables_[f].resize(mipCount_);
        for (uint32_t m = 0; m < mipCount_; m++) {
            upsampleDescTables_[f][m] =
                vk::DescriptorTableBuilder{}
                    .beginDescriptorLayoutSet()
                    .beginDescriptorLayoutSetBinding()
                    .defineDescriptorLayoutSetBinding({
                        .binding = 0,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .defineDescriptorLayoutSetBinding({
                        .binding = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .defineDescriptorLayoutSetBinding({
                        .binding = 2,
                        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        .descriptorCount = 1,
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    })
                    .endDescriptorLayoutSetBinding()
                    .endDescriptorLayoutSet()
                    .definePushConstant(VkPushConstantRange{
                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                        .offset = 0,
                        .size = sizeof(BloomUpsamplePushConstant),
                    })
                    .build(framework->device());

            // Will be bound dynamically during render since src/dst change per upsample pass
        }

        // Composite descriptor table
        compositeDescTables_[f] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet()
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant(VkPushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .offset = 0,
                    .size = sizeof(BloomCompositePushConstant),
                })
                .build(framework->device());

        // Composite: original HDR + bloom mip0 → output
        compositeDescTables_[f]->bindSamplerImageForShader(samplers_[f], hdrInputImages_[f], 0, 0);
        compositeDescTables_[f]->bindSamplerImageForShader(samplers_[f], mipImages_[f][0], 0, 1);
        compositeDescTables_[f]->bindImage(hdrOutputImages_[f], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    }
}

void BloomModule::initPipelines() {
    auto framework = framework_.lock();
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    downsampleShader_ = vk::Shader::create(device, (shaderPath / "world/bloom/bloom_downsample_comp.spv").string());
    downsamplePipeline_ = vk::ComputePipelineBuilder{}
                              .defineShader(downsampleShader_)
                              .definePipelineLayout(downsampleDescTables_[0][0])
                              .build(device);

    upsampleShader_ = vk::Shader::create(device, (shaderPath / "world/bloom/bloom_upsample_comp.spv").string());
    upsamplePipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(upsampleShader_)
                            .definePipelineLayout(upsampleDescTables_[0][0])
                            .build(device);

    compositeShader_ = vk::Shader::create(device, (shaderPath / "world/bloom/bloom_composite_comp.spv").string());
    compositePipeline_ = vk::ComputePipelineBuilder{}
                             .defineShader(compositeShader_)
                             .definePipelineLayout(compositeDescTables_[0])
                             .build(device);
}

BloomModuleContext::BloomModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                       std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                       std::shared_ptr<BloomModule> bloomModule,
                                       uint32_t frameIndex)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      bloomModule(bloomModule),
      frameIndex(frameIndex),
      hdrInputImage(bloomModule->hdrInputImages_[frameIndex]),
      hdrOutputImage(bloomModule->hdrOutputImages_[frameIndex]) {}

void BloomModuleContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = bloomModule.lock();
    auto &mipImages = module->mipImages_[frameIndex];

    // ===== Phase 1: Transition input to SHADER_READ_ONLY =====
    {
        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 srcAccess = 0;
        if (hdrInputImage->imageLayout() != VK_IMAGE_LAYOUT_UNDEFINED) {
            srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                       VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            srcAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        }

        std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers;
        imageBarriers.push_back({
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = hdrInputImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = hdrInputImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });

        // All mip images to GENERAL for storage write
        for (uint32_t m = 0; m < module->mipCount_; m++) {
            imageBarriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                .srcAccessMask = 0,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                .oldLayout = mipImages[m]->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = mipImages[m],
                .subresourceRange = vk::wholeColorSubresourceRange,
            });
            mipImages[m]->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        }

        worldCommandBuffer->barriersBufferImage({}, imageBarriers);
        hdrInputImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // ===== Phase 2: Downsample passes =====
    worldCommandBuffer->bindComputePipeline(module->downsamplePipeline_);

    for (uint32_t m = 0; m < module->mipCount_; m++) {
        auto srcImage = (m == 0) ? hdrInputImage : mipImages[m - 1];
        uint32_t srcW = srcImage->width();
        uint32_t srcH = srcImage->height();
        uint32_t dstW = mipImages[m]->width();
        uint32_t dstH = mipImages[m]->height();

        BloomDownsamplePushConstant pc{};
        pc.srcTexelSizeX = 1.0f / static_cast<float>(srcW);
        pc.srcTexelSizeY = 1.0f / static_cast<float>(srcH);
        pc.threshold = module->threshold_;
        pc.softKnee = module->softKnee_;
        pc.applyThreshold = (m == 0) ? 1 : 0;
        pc.padding0 = 0.0f;
        pc.padding1 = 0.0f;
        pc.padding2 = 0.0f;

        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                           module->downsampleDescTables_[frameIndex][m]->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomDownsamplePushConstant), &pc);

        worldCommandBuffer->bindDescriptorTable(module->downsampleDescTables_[frameIndex][m],
                                                VK_PIPELINE_BIND_POINT_COMPUTE);

        uint32_t groupX = (dstW + 7) / 8;
        uint32_t groupY = (dstH + 7) / 8;
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

        // Barrier: mipImages[m] storage write → sampled read (for next downsample or upsample)
        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = mipImages[m],
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});
    }

    // ===== Phase 3: Upsample passes (bottom to top) =====
    worldCommandBuffer->bindComputePipeline(module->upsamplePipeline_);

    // Start from second-to-last mip, upsample from bottom mip into it
    for (int m = static_cast<int>(module->mipCount_) - 2; m >= 0; m--) {
        auto srcImage = mipImages[m + 1]; // smaller mip (bloom source)
        auto dstImage = mipImages[m];     // larger mip (accumulate into)

        // Bind descriptors dynamically for this upsample pass
        auto &descTable = module->upsampleDescTables_[frameIndex][m];
        descTable->bindSamplerImageForShader(module->samplers_[frameIndex], srcImage, 0, 0);
        descTable->bindImage(dstImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        descTable->bindSamplerImageForShader(module->samplers_[frameIndex], dstImage, 0, 2);

        uint32_t srcW = srcImage->width();
        uint32_t srcH = srcImage->height();
        uint32_t dstW = dstImage->width();
        uint32_t dstH = dstImage->height();

        BloomUpsamplePushConstant pc{};
        pc.srcTexelSizeX = 1.0f / static_cast<float>(srcW);
        pc.srcTexelSizeY = 1.0f / static_cast<float>(srcH);
        pc.bloomRadius = module->radius_;
        pc.padding = 0.0f;

        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                           descTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomUpsamplePushConstant), &pc);

        worldCommandBuffer->bindDescriptorTable(descTable, VK_PIPELINE_BIND_POINT_COMPUTE);

        uint32_t groupX = (dstW + 7) / 8;
        uint32_t groupY = (dstH + 7) / 8;
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

        // Barrier
        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = dstImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});
    }

    // ===== Phase 4: Composite (original HDR + bloom mip0 → output) =====
    {
        // Transition mip[0] to read, output to GENERAL for write
        std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;

        // mip[0] already in GENERAL, need to ensure read access
        barriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = mipImages[0],
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        mipImages[0]->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Output image to GENERAL
        VkPipelineStageFlags2 outSrcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 outSrcAccess = 0;
        if (hdrOutputImage->imageLayout() != VK_IMAGE_LAYOUT_UNDEFINED) {
            outSrcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            outSrcAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        }
        barriers.push_back({
            .srcStageMask = outSrcStage,
            .srcAccessMask = outSrcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout = hdrOutputImage->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = hdrOutputImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        hdrOutputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        worldCommandBuffer->barriersBufferImage({}, barriers);

        BloomCompositePushConstant pc{};
        pc.intensity = module->intensity_;
        pc.padding0 = 0.0f;
        pc.padding1 = 0.0f;
        pc.padding2 = 0.0f;

        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                           module->compositeDescTables_[frameIndex]->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomCompositePushConstant), &pc);

        worldCommandBuffer->bindComputePipeline(module->compositePipeline_);
        worldCommandBuffer->bindDescriptorTable(module->compositeDescTables_[frameIndex],
                                                VK_PIPELINE_BIND_POINT_COMPUTE);

        uint32_t groupX = (module->width_ + 7) / 8;
        uint32_t groupY = (module->height_ + 7) / 8;
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

        // Transition output for next module
        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = hdrOutputImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});
        hdrOutputImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}
